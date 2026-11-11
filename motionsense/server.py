# server.py
import random
import copy
from typing import List, Union, Dict, Optional, Tuple

import torch

from clients import Client, WearableClientInfo, HardwareEnergyModel


class Server:
    # 🌟 核心修改 1：加入了 log_file 参数
    def __init__(self, model, validation_loader, mps_device, log_file="test_accuracy.txt"):
        self.device = mps_device
        self.model = model.to(self.device)
        self.validation_loader = validation_loader
        self.log_file = log_file  # 保存日志文件名

        self.global_prototypes: Dict[int, torch.Tensor] = {}

        self.tier_weights: Dict[str, float] = {
            "FULL": 1.0,
            "PRUNED": 0.7,
            "MSMALL": 0.5,
        }

        self.lambda_pa: float = 0.0
        self.hw = HardwareEnergyModel()

    # ============================================================
    # Energy / Comm estimation (simple)
    # ============================================================
    @staticmethod
    def _estimate_cnn1d_flops_per_sample(profile: Optional[str]) -> float:
        base = 3.0e6
        if profile is None:
            return base
        p = str(profile).upper()
        if p == "PRUNED":
            return base * 0.55
        if p == "MSMALL":
            return base * 0.25
        return base

    @staticmethod
    def _estimate_comm_bytes(profile: Optional[str]) -> int:
        full = 500_000  # ~500KB
        if profile is None:
            return full
        p = str(profile).upper()
        if p == "PRUNED":
            return int(full * 0.55)
        if p == "MSMALL":
            return int(full * 0.25)
        return full

    def estimate_fl_energy_J(self, profile: Optional[str], n_samples: int, epochs_per_client: int) -> float:
        flops_per_sample = self._estimate_cnn1d_flops_per_sample(profile)
        total_flops = flops_per_sample * float(max(1, n_samples)) * float(max(1, epochs_per_client))
        compute_E = self.hw.compute_energy_J(total_flops)

        comm_bytes = self._estimate_comm_bytes(profile)
        comm_bits = float(comm_bytes) * 8.0
        comm_E = self.hw.comm_energy_J(comm_bits)

        return float(compute_E + comm_E)

    # ----------------------------
    # Weighted global-model aggregation (sample-count weighted)
    # ----------------------------
    def average_weights(self, model_list, sample_counts):
        if len(model_list) == 0:
            return
        if sample_counts is None or len(sample_counts) != len(model_list):
            raise ValueError("sample_counts must be provided and match model_list length")

        weights = [float(max(0, int(n))) for n in sample_counts]
        denom = sum(weights)
        if denom <= 0:
            weights = [1.0 for _ in weights]
            denom = float(len(weights))
        weights = [w / denom for w in weights]

        global_dict = self.model.state_dict()
        with torch.no_grad():
            for k, v in global_dict.items():
                if torch.is_floating_point(v) or torch.is_complex(v):
                    agg = None
                    for w, m in zip(weights, model_list):
                        t = m.state_dict()[k].to(dtype=v.dtype, device=v.device)
                        agg = (w * t) if agg is None else (agg + w * t)
                    global_dict[k] = agg
                else:
                    global_dict[k] = model_list[0].state_dict()[k]
        self.model.load_state_dict(global_dict)

    # ----------------------------
    # MSMALL: dequantize int8 payload back to float model for aggregation
    # ----------------------------
    @staticmethod
    @torch.no_grad()
    def _dequantize_model_from_int8_payload(base_model, q_state: Dict[str, torch.Tensor], scales: Dict[str, torch.Tensor]):
        m = copy.deepcopy(base_model)
        sd = m.state_dict()
        for name, q in q_state.items():
            if name not in sd:
                continue
            s = scales.get(name, None)
            if s is None:
                continue
            deq = (q.float() * s.to(q.device).float()).to(sd[name].dtype)
            sd[name].copy_(deq)
        m.load_state_dict(sd)
        return m

    # ============================================================
    # BACS selection
    # ============================================================
    def _select_clients_bacs(
        self,
        clients: List[WearableClientInfo],
        clients_per_round: int,
        epochs_per_client: int,
        strict_mode: bool = True,
    ) -> List[WearableClientInfo]:
        eligible: List[WearableClientInfo] = []
        for c in clients:
            try:
                profile = c.select_profile().value
            except Exception:
                profile = "FULL"

            n_samples = len(c.loader.dataset) if hasattr(c.loader, "dataset") else 1
            _ = self.estimate_fl_energy_J(profile, n_samples=n_samples, epochs_per_client=epochs_per_client)

            avail_flag = c.availability()
            if strict_mode:
                if avail_flag == 1:
                    eligible.append(c)
            else:
                if c.battery > 0.0:
                    eligible.append(c)

        if len(eligible) <= clients_per_round:
            return eligible
        return random.sample(eligible, clients_per_round)

    # ============================================================
    # Prototype aggregation (tier-aware)
    # ============================================================
    @torch.no_grad()
    def _aggregate_prototypes_tier_aware(self, uploads: List[Tuple[str, Dict[int, torch.Tensor], Dict[int, int]]]) -> Dict[int, torch.Tensor]:
        if len(uploads) == 0:
            return self.global_prototypes

        num: Dict[int, torch.Tensor] = {}
        den: Dict[int, float] = {}

        for prof, p_dict, c_dict in uploads:
            w = float(self.tier_weights.get(str(prof).upper(), 1.0))
            for cls, p in p_dict.items():
                n_c = int(c_dict.get(cls, 0))
                if n_c <= 0:
                    continue
                if cls not in num:
                    num[cls] = (w * float(n_c)) * p.detach().cpu()
                    den[cls] = (w * float(n_c))
                else:
                    num[cls] = num[cls] + (w * float(n_c)) * p.detach().cpu()
                    den[cls] = den[cls] + (w * float(n_c))

        aggregated: Dict[int, torch.Tensor] = {}
        for cls, v in num.items():
            d = float(den.get(cls, 0.0))
            if d > 0.0:
                aggregated[cls] = (v / d).detach().cpu()

        for cls, p_prev in self.global_prototypes.items():
            if cls not in aggregated:
                aggregated[cls] = p_prev.detach().cpu()

        return aggregated

    # ============================================================
    # Federated Learning Loop
    # ============================================================
    def federated_learning(
        self,
        client_loaders_or_infos: Union[List[object], List[WearableClientInfo]],
        num_rounds=5,
        clients_per_round=10,
        epochs_per_client=2,
        use_bacs: bool = False,
        fl_cost_per_round: float = 0.01,  # kept for backward compatibility (ignored)
        bacs_strict_mode: bool = True,
        use_tier_proto: bool = True,
        lambda_pa: float = 0.0,
    ):
        if len(client_loaders_or_infos) == 0:
            raise ValueError("client list is empty")

        is_wearable_infos = isinstance(client_loaders_or_infos[0], WearableClientInfo)
        self.lambda_pa = float(lambda_pa)

        for round_idx in range(num_rounds):
            if use_bacs and is_wearable_infos:
                clients: List[WearableClientInfo] = client_loaders_or_infos  # type: ignore
                selected_infos = self._select_clients_bacs(
                    clients, clients_per_round, epochs_per_client=epochs_per_client, strict_mode=bacs_strict_mode
                )
                selected_loaders = [c.loader for c in selected_infos]
            else:
                client_loaders: List[object] = client_loaders_or_infos  # type: ignore
                selected_loaders = random.sample(client_loaders, clients_per_round)
                selected_infos = None

            if use_bacs and is_wearable_infos and len(selected_loaders) == 0:
                print(f"[Warning] Round {round_idx + 1}: no available clients under strict BACS. Deferring training this round.")
                # 🌟 核心修改 2：使用 self.log_file 动态写入
                with open(self.log_file, mode="a") as f:
                    f.write(f"Round {round_idx + 1} deferred (no available clients)\n")
                for c in client_loaders_or_infos:  # type: ignore
                    c.env_step(fl_selected=False, fl_energy_J=0.0)
                continue

            models = []
            sample_counts = []
            proto_uploads: List[Tuple[str, Dict[int, torch.Tensor], Dict[int, int]]] = []
            selected_energy_by_id: Dict[int, float] = {}

            for i, loader in enumerate(selected_loaders):
                profile = None
                if use_bacs and is_wearable_infos and selected_infos is not None:
                    profile = selected_infos[i].select_profile().value

                n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 1
                fl_E = self.estimate_fl_energy_J(profile, n_samples=n_samples, epochs_per_client=epochs_per_client)
                if use_bacs and is_wearable_infos and selected_infos is not None:
                    selected_energy_by_id[id(selected_infos[i])] = float(fl_E)

                downlink_proto = self.global_prototypes if (use_tier_proto and len(self.global_prototypes) > 0) else None
                lam = self.lambda_pa if (use_tier_proto and profile is not None and str(profile).upper() in ("MSMALL",)) else 0.0

                client = Client(
                    self.model,
                    loader,
                    mps_device=self.device,
                    profile=profile,
                    global_prototypes=downlink_proto,
                    lambda_pa=lam,
                )
                local_model = client.train(epochs=epochs_per_client)

                if (
                    profile is not None
                    and str(profile).upper() in ("MSMALL",)
                    and getattr(client, "int8_state_dict", None) is not None
                    and getattr(client, "int8_scales", None) is not None
                ):
                    try:
                        print(f"[MSMALL] INT8 payload bytes: {getattr(client, 'int8_num_bytes', 0)}")
                        local_model_for_agg = self._dequantize_model_from_int8_payload(
                            local_model, client.int8_state_dict, client.int8_scales
                        )
                        models.append(local_model_for_agg)
                        sample_counts.append(n_samples)
                    except Exception:
                        models.append(local_model)
                        sample_counts.append(n_samples)
                else:
                    models.append(local_model)
                    sample_counts.append(n_samples)

                if use_tier_proto:
                    prof_str = str(profile).upper() if profile is not None else "FULL"
                    proto_uploads.append((prof_str, client.proto_dict, client.count_dict))

            self.average_weights(models, sample_counts)
            self.evaluate()

            if use_tier_proto:
                self.global_prototypes = self._aggregate_prototypes_tier_aware(proto_uploads)

            if use_bacs and is_wearable_infos and selected_infos is not None:
                selected_set = set(id(c) for c in selected_infos)
                for c in client_loaders_or_infos:  # type: ignore
                    fl_selected = (id(c) in selected_set)
                    fl_E = float(selected_energy_by_id.get(id(c), 0.0)) if fl_selected else 0.0
                    c.env_step(fl_selected=fl_selected, fl_energy_J=fl_E)

            # 🌟 核心修改 3：使用 self.log_file 动态写入
            with open(self.log_file, mode="a") as f:
                print(f"Round {round_idx + 1} completed")
                f.write(f"Round {round_idx + 1} completed\n")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()

        acc = 100.0 * correct / max(1, total)
        print(f"Validation Accuracy: {acc:.6f}%")
        # 🌟 核心修改 4：使用 self.log_file 动态写入
        with open(self.log_file, mode="a") as f:
            f.write(f"Validation Accuracy: {acc:.6f}%\n")