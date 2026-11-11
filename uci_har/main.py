import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional

from model import UCIHARModel
from server import Server
from uci_har_datast import UCIHARDataset

from clients import WearableClientInfo, BACSDecisionEngine, EnergyParams, UserState


def split_data_with_shared_and_specialty(
        train_dataset,
        num_edges: int = 50,
        vehicles_per_edge: int = 1,
        alpha: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 0,
        shared_data_ratio: float = 0.37,
        seed: int = 42
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_clients = num_edges * vehicles_per_edge

    if not hasattr(train_dataset, "y"):
        raise AttributeError("dataset  dataset.yLongTensor")

    y = train_dataset.y.cpu().numpy()
    n = len(train_dataset)
    num_classes = int(y.max() + 1)

    class_indices = [np.where(y == k)[0] for k in range(num_classes)]
    shared_indices = []
    specialty_indices = []

    for k in range(num_classes):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        split_point = int(len(idx_k) * shared_data_ratio)
        shared_indices.extend(idx_k[:split_point])
        specialty_indices.extend(idx_k[split_point:])

    np.random.shuffle(shared_indices)
    client_shared_splits = np.array_split(shared_indices, num_clients)

    client_specialty_splits = [[] for _ in range(num_clients)]
    specialty_y = y[specialty_indices]
    specialty_class_indices = [np.where(specialty_y == k)[0] for k in range(num_classes)]

    for k in range(num_classes):
        idx_k = specialty_class_indices[k]
        if len(idx_k) == 0:
            continue

        real_idx_k = np.array(specialty_indices)[idx_k]
        np.random.shuffle(real_idx_k)

        proportions = np.random.dirichlet([alpha] * num_clients)

        totals = np.array([len(ci) for ci in client_specialty_splits], dtype=np.float64)
        if totals.sum() > 0:
            reweight = 1.0 / (totals + 1.0)
            proportions = proportions * reweight

        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(real_idx_k)).astype(int)[:-1]
        splits = np.split(real_idx_k, split_points)

        for client_id in range(num_clients):
            client_specialty_splits[client_id].extend(splits[client_id].tolist())

    client_loades = []
    for cid in range(num_clients):
        combined_indices = list(client_shared_splits[cid]) + client_specialty_splits[cid]
        np.random.shuffle(combined_indices)

        subset = Subset(train_dataset, combined_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False
        )
        client_loaders.append(loader)

    print(f"\n[Shared+Specialty Split] alpha={alpha}, shared_ratio={shared_data_ratio}, clients={num_clients}")
    for cid in range(min(num_clients, 10)):
        idx = np.array(list(client_shared_splits[cid]) + client_specialty_splits[cid], dtype=np.int64)
        hist = np.bincount(y[idx], minlength=num_classes)
        print(f"  Client {cid:02d}: n={len(idx):4d}, class_hist={hist.tolist()}")
    if num_clients > 10:
        print("  ... ( 10 )\n")

    return client_loaders


def build_wearable_clients_for_bacs(
        client_loaders,
        seed: int = 42,
        tau: int = 6,
        theta_eas: float = 0.0,
        B_min: float = 0.15,
        alpha_eas: float = 1.0,
        beta_eas: float = 1.0,
        consumption_mode: str = "mean",
):
    rng = nprandom.default_rng(seed)

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
            enegy_params=energy_params,
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

    data_dir = "./data/UCI-HAR Dataset"
    train_X_file = os.path.join(data_dir, "train", "X_train.txt")
    train_y_file = os.path.join(data_dir, "train", "y_train.txt")
    test_X_file = os.path.join(data_dir, "test", "X_test.txt")
    test_y_file = os.path.join(data_dir, "test", "y_test.txt")

    train_dataset = UCIHARDataset(train_X_file, train_y_file)
    test_dataset = UCIHARDataset(test_X_file, test_y_file)

    batch_size = 32
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    client_loaders = split_data_with_shared_and_specialty(
        train_dataset,
        num_edges=50,
        vehicles_per_edge=1,
        alpha=5,
        batch_size=batch_size,
        num_workers=0,
        shared_data_ratio=0.4,
        seed=42
    )

    global_model = UCIHARModel(input_dim=561, hidden_dims=[256, 128, 64], number_of_classes=6, p=0.2)
    server = Server(global_model, validation_loader, device)

    
    alpha_c = 5.0

    wearable_clients = build_wearable_clients_for_bacs(
        client_loaders,
        seed=42,
        tau=6,
        theta_eas=0.0,
        B_min=0.15,
        alpha_eas=1.0,
        beta_eas=1.0,
        consumption_mode="mean"
    )

    print("\n==================================================")
    print("🚀 Starting Elastic-AFL Main Framework Training on UCI")
    print("==================================================")

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
        alpha_c=alpha_c,
    )


if __name__ == "__main__":
    main()
