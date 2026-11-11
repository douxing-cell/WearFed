
import os
import copy
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from model import UCHARModel
from uci_har_dataset import UCIHARDataset


PROFILE_COSTS = {
    "FULL": {"size_mb": 0.75, "comp_energy": 5.0},
    "PRUNED_M": {"size_mb": 0.38, "comp_energy": 2.5},
    "PRUNED_S": {"size_mb": 0.20, "comp_energy": 1.2},
}


def calculate_round_metrics(profile, epochs):
    cost = PROFILE_COSTS.get(profile, PROFILE_COSTS["FULL"])
    size_mb = cost["size_mb"]
    comm_mb = size_mb * 2.0
    time_s = size_mb / 1.25
    comm_energy_j = (time_s * 1.0) + (time_s * 2.0)
    comp_energy_j = cost["comp_energy"] * epochs
    return comm_mb, comp_energy_j, comm_energy_j


def create_federated_loaders_dirichlet_noniid(dataset, num_clients=50, batch_size=32, dirichlet_alpha=1.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    y = dataset.y.cpu().numpy()
    num_classes = int(y.max() + 1)
    class_indices = [np.where(y == k)[0] for k in range(num_classes)]
    for k in range(num_classes): np.random.shuffle(class_indices[k])

    for _ in range(50)
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = class_indices[k]
            if len(idx_k) == 0: continue
            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
            totals = np.array([len(ci) for ci in client_indices], dtype=np.float64)
            if totals.sum() > 0:
                proportions = proportions * (1.0 / (totals + 1.0))
                proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, split_points)
            for cid in range(num_clients):
                client_indices[cid].extend(splits[cid].tolist())
        if min([len(ci) for ci in client_indices]) >= 2: break

    loaders = []
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])
        loaders.append(DataLoader(Subset(dataset, client_indices[cid]), batch_size=batch_size, shuffle=True))
    return loaders


class ClientEFLCF:
    def __init__(self, model, dataloader, mps_device, profile="FULL"):
        self.device = ms_device
        self.model = copy.deepcopy(model).to(self.device)
        self.dataloader = dataloader
        if hasattr(self.model, "set_profile"):
            self.model.set_profile(profile)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.2)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                loss = self.criterion(self.model(data), target)
                loss.backward()
                self.optimizer.step()
        return self.model

class ServerEFLCF:
    def __init__(self, model, validation_loader, mps_device):
        self.device = mps_device
        self.model = model.to(self.device)
        self.validation_loader = validation_loader
        self.total_comm_mb = 0.0
        self.total_comp_energy = 0.0
        self.total_comm_energy = 0.0

    def average_weights(self, model_list, sample_counts):
        weights = [float(max(0, int(n))) for n in sample_counts]
        denom = sum(weghts)
        weights = [w / denom for w in weights] if denom > 0 else [1.0] * len(weights)
        global_dict = self.model.state_dict()
        with torch.no_grad():
            for k, v in global_dict.items():
                if torch.is_floating_point(v) or torch.is_complex(v):
                    agg = None
                    for w, m in zip(weights, model_list):
                        t = m.state_dict()[k].to(dtype=v.dtype, device=v.device)
                        agg = w * t if agg is None else agg + (w * t)
                    global_dict[k] = agg
        self.model.load_state_dict(global_dict)

    def federated_learning_eflcf(self, client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=1):
        for round_idx in range(num_rounds):
            selected_indices = random.sample(range(len(client_loaders)), clients_per_round)
            models, sample_counts = [], []

            for cid in selected_indices:
                loader= client_loaders[cid]
                constraint_state = random.random()

                
                if constraint_state > 0.6:
                    assigned_profile = "FULL"
                elif constraint_state > 0.2:
                    assigned_profile = "PRUNED_M"
                else:
                    assigned_profile = "PRUNED_S"

                client = ClientEFLCF(self.model, loader, self.device, profile=assigned_profile)
                models.append(client.train(epochs=epochs_per_client))
                sample_counts.append(len(loader.dataset))

                comm, comp_e, comm_e = calculate_round_metrics(assigned_profile, epochs_per_client)
                self.total_comm_mb += comm
                self.total_comp_energy += comp_e
                self.total_comm_energy += comm_e

            self.average_weights(models, sample_counts)
            self.model.eval()
            if hasattr(self.model, "set_profile"):
                self.model.set_profile("FULL")

            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.validation_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.numel()

            acc = 100.0 * correct / max(1, total)
            print(
                f"Round {round_idx + 1} | Acc: {acc:.2f}% | Comm: {self.total_comm_mb:.2f}MB | CompEng: {self.total_comp_energy:.2f}J | CommEng: {self.total_comm_energy:.2f}J")

            with open("test_accuracy_eflcf.txt", mode="a") as f:
                f.write(
                    f"Round {round_idx + 1} | Acc: {acc:.4f}% | Comm(MB): {self.total_comm_mb:.2f} | Comp(J): {self.total_comp_energy:.2f} | CommEng(J): {self.total_comm_energy:.2f}\n")


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data/UCI-HAR Dataset"
    train_dataset = UCIHARDataset(os.path.join(data_dir, "train", "X_train.txt"),
                                  os.path.join(data_dir, "train", "y_train.txt"))
    test_dataset = UCIHARDataset(os.path.join(data_dir, "test", "X_test.txt"),
                                 os.path.join(data_dir, "test", "y_test.txt"))
    validation_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(train_dataset, num_clients=50, batch_size=32,
                                                               dirichlet_alpha=5, seed=42)
    global_model = UCIHARModel(input_dim=561, hidden_dims=[256, 128, 64], number_of_classes=6, p=0.2)

    
    global_model._profile_ratios["PRUNED_M"] = 0.7
    global_model._profile_ratios["PRUNED_S"] = 0.5

    server = ServerEFLCF(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting EFLCF Baseline Training (Fair Float32 Mode)")
    print("==================================================")
    server.federated_learning_eflcf(client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=1)


if __name__ == "__main__":
    main()
