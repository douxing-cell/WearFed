import copy
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import optim, nn


class Client:
    def __init__(
            self,
            model,
            dataloader,
            mps_device,
            profile: Optional[str] = None,
            global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
            lambda_pa: float = 0.0,
    ):
        self.device = mps_device
        self.model = copy.deepcopy(model)

        self.profile: Optional[str] = None
        if profile is not None and hasattr(self.model, "set_profile"):
            try:
                self.model.set_profile(profile)
                self.profile = str(profile).upper()
            except Exception:
                self.profile = None

        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.2)
        self.criterion = nn.CrossEntropyLoss()

        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = global_prototypes
        self.lambda_pa: float = float(lambda_pa)

        self.proto_dict: Dict[int, torch.Tensor] = {}
        self.count_dict: Dict[int, int] = {}

        self.int8_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.int8_scales: Optional[Dict[str, torch.Tensor]] = None
        self.int8_num_bytes: int = 0

    def _prototype_alignment_loss(
            self,
            feats: torch.Tensor,
            targets: torch.Tensor,
            global_prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return feats.sum() * 0.0

        proto_list = []
        valid_mask = []
        for y in targets.tolist():
            if int(y) in global_prototypes:
                proto_list.append(global_prototypes[int(y)].to(feats.device).view(1, -1))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            return feats.sum() * 0.0

        feats_valid = feats[torch.tensor(valid_mask, device=feats.device)]
        protos = torch.cat(proto_list, dim=0)

        # 修复：采用维度求均值（mean）替代求和（sum），消除高维特征空间带来的数值爆炸
        dist_sq = torch.mean((feats_valid - protos) ** 2, dim=1)
        return (0.5 * dist_sq).mean()

    @torch.no_grad()
    def compute_prototypes(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        if not hasattr(self.model, "extract_features"):
            return {}, {}

        self.model.eval()
        sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        for data, target in self.dataloader:
            data = data.to(self.device)
            target = target.to(self.device)

            feats = self.model.extract_features(data)
            for i in range(feats.shape[0]):
                y = int(target[i].item())
                f = feats[i].detach()
                if y not in sums:
                    sums[y] = f.clone()
                    counts[y] = 1
                else:
                    sums[y] += f
                    counts[y] += 1

        proto: Dict[int, torch.Tensor] = {}
        for y, s in sums.items():
            n = max(1, int(counts.get(y, 0)))
            proto[y] = (s / float(n)).detach().cpu()

        return proto, counts

    def train(self, epochs=1, data_ratio=1.0):
        self.model.train()

        # 修复：使用动态随机子采样机制保障样本分布一致性，避免顺序截断
        if data_ratio < 1.0:
            dataset = self.dataloader.dataset
            total_samples = len(dataset)
            target_samples = max(1, int(total_samples * float(data_ratio)))

            indices = torch.randperm(total_samples)[:target_samples].tolist()
            subset = torch.utils.data.Subset(dataset, indices)

            active_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=self.dataloader.batch_size,
                shuffle=True,
                drop_last=False
            )
        else:
            active_loader = self.dataloader

        for _ in range(epochs):
            for data, target in active_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.model(data)
                loss = self.criterion(logits, target)

                if (
                        self.lambda_pa > 0.0
                        and self.global_prototypes is not None
                        and hasattr(self.model, "extract_features")
                        and (self.profile is not None and self.profile.upper() in ("MSMALL",))
                ):
                    feats = self.model.extract_features(data)
                    pa = self._prototype_alignment_loss(feats, target, self.global_prototypes)
                    loss = loss + self.lambda_pa * pa

                loss.backward()

                # 修复：引入全局梯度裁剪，作为防御局部过度拟合与量级突变的底层安全防线
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

        self.proto_dict, self.count_dict = self.compute_prototypes()

        if hasattr(self.model, "export_int8_state_dict") and (
                self.profile is not None and self.profile.upper() in ("MSMALL",)):
            try:
                q_state, scales = self.model.export_int8_state_dict()
                self.int8_state_dict = q_state
                self.int8_scales = scales
                if hasattr(self.model, "estimate_int8_payload_bytes"):
                    self.int8_num_bytes = int(self.model.estimate_int8_payload_bytes(q_state, scales))
            except Exception:
                self.int8_state_dict, self.int8_scales, self.int8_num_bytes = None, None, 0

        return self.model


class UserState(IntEnum):
    IDLE = 0
    ACTIVE = 1
    CHARGING = 2


@dataclass
class EnergyParams:
    charge_gain: Dict[UserState, float]
    consume_cost: Dict[UserState, float]

    @staticmethod
    def default() -> "EnergyParams":
        return EnergyParams(
            charge_gain={
                UserState.IDLE: 0.0,
                UserState.ACTIVE: 0.0,
                UserState.CHARGING: 0.06,
            },
            consume_cost={
                UserState.IDLE: 0.004,
                UserState.ACTIVE: 0.012,
                UserState.CHARGING: 0.001,
            },
        )


class ModelProfile(str, Enum):
    FULL = "FULL"
    PRUNED = "PRUNED"
    MSMALL = "MSMALL"


@dataclass
class ProfileLookupParams:
    eas_high: float = 0.50
    eas_mid: float = 0.30
    cpu_low: float = 0.35


def lookup_profile(
        eas_value: float,
        cpu_util: float,
        s_t: UserState,
        params: ProfileLookupParams = ProfileLookupParams(),
) -> ModelProfile:
    if s_t == UserState.CHARGING:
        return ModelProfile.FULL

    if eas_value >= params.eas_high and cpu_util <= params.cpu_low:
        return ModelProfile.FULL
    if eas_value >= params.eas_mid:
        return ModelProfile.PRUNED

    return ModelProfile.MSMALL


class BACSDecisionEngine:
    def __init__(
            self,
            state_transition_P: np.ndarray,
            energy_params: EnergyParams,
            tau: int = 6,
            theta_eas: float = 0.0,
            B_min: float = 0.15,
            alpha_eas: float = 1.0,
            beta_eas: float = 1.0,
            consumption_mode: str = "mean",
            rng: Optional[np.random.Generator] = None,
            pred_error: float = 0.03,
    ):
        self.P = state_transition_P.astype(np.float64)
        self.energy = energy_params
        self.tau = int(tau)
        self.theta_eas = float(theta_eas)
        self.B_min = float(B_min)
        self.alpha_eas = float(alpha_eas)
        self.beta_eas = float(beta_eas)
        self.consumption_mode = consumption_mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pred_error = float(pred_error)

    def _state_dist_k(self, s_t: UserState, k: int) -> np.ndarray:
        Pk = np.linalg.matrix_power(self.P, k)
        return Pk[int(s_t), :]

    def expected_gain_and_consumption_series(self, s_t: UserState) -> Tuple[float, List[float]]:
        gain_sum = 0.0
        cons_list: List[float] = []

        for k in range(1, self.tau + 1):
            dist = self._state_dist_k(s_t, k)

            e_gain = (
                    dist[UserState.IDLE] * self.energy.charge_gain[UserState.IDLE]
                    + dist[UserState.ACTIVE] * self.energy.charge_gain[UserState.ACTIVE]
                    + dist[UserState.CHARGING] * self.energy.charge_gain[UserState.CHARGING]
            )
            e_cons = (
                    dist[UserState.IDLE] * self.energy.consume_cost[UserState.IDLE]
                    + dist[UserState.ACTIVE] * self.energy.consume_cost[UserState.ACTIVE]
                    + dist[UserState.CHARGING] * self.energy.consume_cost[UserState.CHARGING]
            )

            gain_sum += float(e_gain)
            cons_list.append(float(e_cons))

        return gain_sum, cons_list

    def eas(self, B_cur: float, s_t: UserState) -> float:
        gain_sum, cons_list = self.expected_gain_and_consumption_series(s_t)
        cons_term = float(sum(cons_list)) if cons_list else 0.0
        return float(B_cur + self.alpha_eas * gain_sum - self.beta_eas * cons_term)

    def availability_flag(self, B_cur: float, s_t: UserState) -> int:
        if s_t == UserState.CHARGING:
            return 1
        if float(B_cur) < self.B_min:
            return 0
        return 1 if self.eas(B_cur, s_t) >= self.theta_eas else 0

    def availability_flag_formula8(self, B_cur: float, s_t: UserState, fl_cost: float) -> int:
        if s_t == UserState.CHARGING:
            return 1

        B_cur = float(B_cur)
        if B_cur < self.B_min:
            return 0

        if float(self.eas(B_cur, s_t)) < float(self.theta_eas):
            return 0

        projected = self.step_battery(B_cur, s_t, fl_selected=True, fl_cost=float(fl_cost))
        return 1 if float(projected) >= (float(self.B_min) + float(self.pred_error)) else 0

    def sample_next_state(self, s_t: UserState) -> UserState:
        probs = self.P[int(s_t), :]
        nxt = int(self.rng.choice([0, 1, 2], p=probs))
        return UserState(nxt)

    def step_battery(self, B_t: float, s_t: UserState, fl_selected: bool, fl_cost: float) -> float:
        B_next = float(B_t)
        B_next += float(self.energy.charge_gain[s_t])
        B_next -= float(self.energy.consume_cost[s_t])
        if fl_selected:
            B_next -= float(fl_cost)
        return float(np.clip(B_next, 0.0, 1.0))


@dataclass
class WearableClientInfo:
    loader: object
    battery: float
    state: UserState
    bacs: BACSDecisionEngine
    cpu_util: float = 0.0
    profile_params: ProfileLookupParams = field(default_factory=ProfileLookupParams)

    def availability(self) -> int:
        return self.bacs.availability_flag(self.battery, self.state)

    def availability_formula8(self, fl_cost: float) -> int:
        if hasattr(self.bacs, "availability_flag_formula8"):
            return self.bacs.availability_flag_formula8(self.battery, self.state, float(fl_cost))
        return self.availability()

    def select_profile(self) -> ModelProfile:
        eas_val = self.bacs.eas(self.battery, self.state)
        return lookup_profile(eas_val, float(self.cpu_util), self.state, self.profile_params)

    def _sample_cpu_util(self) -> float:
        try:
            if self.state == UserState.ACTIVE:
                return float(self.bacs.rng.uniform(0.4, 0.9))
            elif self.state == UserState.CHARGING:
                return float(self.bacs.rng.uniform(0.0, 0.2))
            else:
                return float(self.bacs.rng.uniform(0.0, 0.3))
        except Exception:
            return float(self.cpu_util)

    def env_step(self, fl_selected: bool, fl_cost: float) -> None:
        gamma = 1.0
        if self.state == UserState.ACTIVE:
            gamma = 1.5
        elif self.state == UserState.IDLE:
            gamma = 1.0
        elif self.state == UserState.CHARGING:
            gamma = 1.0

        actual_fl_cost = float(fl_cost) * float(gamma)

        self.battery = self.bacs.step_battery(
            self.battery,
            self.state,
            fl_selected=fl_selected,
            fl_cost=actual_fl_cost,
        )

        self.state = self.bacs.sample_next_state(self.state)
        self.cpu_util = self._sample_cpu_util()