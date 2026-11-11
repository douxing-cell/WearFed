
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional

from model import CNN_V3_V4
from server import Server
from motionsense_dataset import MotionSenseDataset





def create_federated_loaders_dirichlet_noniid(
        dataset,
        num_clients: int = 50
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

    y = dataset.y.cpu().numpy()
    num_classes = int(y.max() + 1)
    class_indices = [np.where(y == k)[0] for k in range(num_classes)]
    for k in range(num_classes):
        np.random.shuffle(class_indices[k])

    client_indices = None
    for _ in range(max_retry):
        client_indices = [[ for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = class_indices[k]
            if len(idx_k) == 0: continue
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

        if min([len(ci) for ci in client_indices]) >= min_samples_per_client:
            break
        client_indices = None

    client_loaders = []
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])
        subset = Subset(dataset, client_indices[cid])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    if verbose:
        print(f"\n[FedAvg Baseline] Dirichlet Alpha={dirichlet_alpha}, Clients={num_clients}")
    return client_loaders





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data"  
    window_size = 128
    step_size = 64
    batch_size = 32

    print("Loading MotionSense train (FedAvg)...")
    train_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=True)

    print("Loading MotionSense val/test (FedAvg)...")
    test_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=False,
                                      scaler=train_dataset.scaler)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
    
    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset, num_clients=50, batch_size=batch_size, dirichlet_alpha=5, min_samples_per_client=10, seed=42,
        verbose=False
    )

    
    global_model = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=6, p=0.2, window_size=window_size,
                             num_features=12)

    
    server = Server(global_model, validation_loader, device, log_file="test_accuracy_fedavg.txt")

    
    print("\n==================================================")
    print("🚀 Starting Traditional FedAvg Training")
    print("==================================================")

    
    server.federated_learning(
        client_loaders_or_infos=client_loaders,  
        num_rounds=50,
        clients_per_round=10,
        epochs_per_client=1,
        use_bacs=False,  
        use_tier_proto=False,  
        lambda_pa=0.0,  
    )


if __name__ == "__main__":
    main()
