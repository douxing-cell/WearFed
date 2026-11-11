# main.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional

from model import CNN_V3_V4
from server import Server
from motionsense_dataset import MotionSenseDataset

from clients import WearableClientInfo, BACSDecisionEngine, EnergyParams, UserState


def create_federated_loaders_dirichlet_noniid(
    dataset,
    num_clients: int = 20,
    batch_size: int = 32,
    dirichlet_alpha: float = 5,
    min_samples_per_client: int = 10,
    max_retry: int = 50,
    seed: Optional[int] = 42,
    verbose: bool = True,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not hasattr(dataset, "y"):
        raise AttributeError("dataset 必须包含标签张量属性 dataset.y（LongTensor）。")

    y = dataset.y.cpu().numpy()
    n = len(dataset)
    num_classes = int(y.max() + 1)

    if n < num_clients * min_samples_per_client:
        raise ValueError(
            f"样本总数 {n} < num_clients*min_samples_per_client={num_clients*min_samples_per_client}，"
            f"请减小 num_clients 或 min_samples_per_client。"
        )

    class_indices = [np.where(y == k)[0] for k in range(num_classes)]
    for k in range(num_classes):
        np.random.shuffle(class_indices[k])

    client_indices = None
    for _ in range(max_retry):
        client_indices = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = class_indices[k]
            if len(idx_k) == 0:
                continue

            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)

            totals = np.array([len(ci) for ci in client_indices], dtype=np.float64)
            if totals.sum() > 0:
                reweight = 1.0 / (totals + 1.0)
                proportions = proportions * reweight
                proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, split_points)

            for client_id in range(num_clients):
                client_indices[client_id].extend(splits[client_id].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_samples_per_client:
            break
        client_indices = None

    if client_indices is None:
        raise RuntimeError(
            f"在 {max_retry} 次重采样后仍无法满足 min_samples_per_client={min_samples_per_client}。"
            f"建议：增大 alpha 或减小 num_clients/min_samples_per_client。"
        )

    client_loaders = []
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])
        subset = Subset(dataset, client_indices[cid])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    if verbose:
        print(f"\n[Dirichlet non-IID split] alpha={dirichlet_alpha}, clients={num_clients}, classes={num_classes}")
        for cid in range(min(num_clients, 10)):
            idx = np.array(client_indices[cid], dtype=np.int64)
            hist = np.bincount(y[idx], minlength=num_classes)
            print(f"  Client {cid:02d}: n={len(idx):4d}, class_hist={hist.tolist()}")
        if num_clients > 10:
            print("  ... (仅展示前 10 个客户端统计)\n")

    return client_loaders


def build_wearable_clients_for_bacs(
    client_loaders,
    seed: int = 42,
    tau: int = 6,
    theta_eas: float = 0.0,
    B_min: float = 0.15,
    alpha_eas: float = 0.5,
    beta_eas: float = 0.5,
    consumption_mode: str = "mean",
):
    rng = np.random.default_rng(seed)

    P = np.array([
        [0.70, 0.20, 0.10],
        [0.35, 0.55, 0.10],
        [0.20, 0.10, 0.70],
    ], dtype=np.float64)

    energy_params = EnergyParams.default()

    wearable_clients = []
    for loader in client_loaders:
        battery = float(rng.uniform(B_min, 1.0))
        state = UserState(int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1])))

        engine = BACSDecisionEngine(
            state_transition_P=P,
            energy_params=energy_params,
            tau=tau,
            theta_eas=theta_eas,
            B_min=B_min,
            alpha_eas=alpha_eas,
            beta_eas=beta_eas,
            consumption_mode=consumption_mode,
            rng=rng,
        )

        wearable_clients.append(
            WearableClientInfo(loader=loader, battery=battery, state=state, bacs=engine)
        )

    return wearable_clients


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data"       # 你的 MotionSense 根目录（包含 A_DeviceMotion_data）
    window_size = 128
    step_size = 64

    print("Loading MotionSense train...")
    train_dataset = MotionSenseDataset(
        data_dir=data_dir,
        window_size=window_size,
        step_size=step_size,
        train=True,
    )

    print("Loading MotionSense val/test...")
    test_dataset = MotionSenseDataset(
        data_dir=data_dir,
        window_size=window_size,
        step_size=step_size,
        train=False,
        scaler=train_dataset.scaler,
    )

    batch_size = 32
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset,
        num_clients=50,
        batch_size=batch_size,
        dirichlet_alpha=0.5,
        min_samples_per_client=10,
        seed=42,
        verbose=True,
    )

    global_model = CNN_V3_V4(
        out_1=32,
        out_2=64,
        out_3=128,
        number_of_classes=6,
        p=0.2,
        window_size=window_size,
        num_features=12,
    )
    server = Server(global_model, validation_loader, device)

    wearable_clients = build_wearable_clients_for_bacs(
        client_loaders,
        seed=42,
        tau=6,
        theta_eas=0.0,
        B_min=0.15,
        alpha_eas=0.5,
        beta_eas=0.5,
        consumption_mode="mean"
    )

    server.federated_learning(
        wearable_clients,
        num_rounds=50,
        clients_per_round=10,
        epochs_per_client=1,
        use_bacs=True,
        fl_cost_per_round=0.01,
        bacs_strict_mode=True,
        use_tier_proto=True,
        lambda_pa=0.1,
    )


if __name__ == "__main__":
    main()
