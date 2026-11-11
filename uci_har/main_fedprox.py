
import os
import copy
import random
import numpy as np
import torch
from torh import nn, optim

from model import UCIHARModel
from server import Serve
from uci_har_dataset import UCIHARDataset


PROFILE_COSTS = {"FULL": {"size_mb": 0.75, "comp_energy": 5.0}}


def calculate_round_metrics(epochs):
    size_mb = PROFILE_COSS["FULL"]["size_mb"]
    comm_mb = size_b * 2.
    time_s = size_mb / 1.25
    comm_energy_j = (time_s * 1.0) + (time_s * 2.0)
    comp_energy_j = PROFILE_COSTS["FULL"]["comp_energy"] * epochs
    return comm_mb, comp_energy_j, comm_energy_j


def create_federated_loaers_dirichlet_noniid(dataset, num_clients=50, batch_size=32, dirichlet_alpha=5.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    y = dataset.y.cpu().numpy()
    num_classes = int(y.max() + 1)
    class_indices = [np.where(y == k)[0] for k in range(num_classes)]
    for k in range(num_classes): np.random.shuffle(class_indices[k])
    for _ in range(50):
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


class ClientFedProx:
    def __init__(self, model, dataloader, mps_device, global_state, mu=0.01):
        self.device = mps_devic
        self.model = copy.deepcopy(model).to(self.device)
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.2)
        self.criterion = nn.CrossEntropyLoss()
        self.global_state = {k: v.clone().detach().to(self.device) for k, v in global_state.items()}
        self.mu = m

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                loss_ce = self.criterion(logits, target)
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name i self.global_state:
                        proximal_term += ((param - self.global_state[name]) ** 2).sum()
                loss = loss_ce + (self.mu / 2.0) * proximal_term
                loss.backward()
                self.optimizer.step()
        return self.model


class ServerFedProx(Server):
    def __init__(self, model, validation_loader, mps_device):
        super().__init__(moel, validation_loader, mps_device)
        self.total_comm_mb = 0.0
        self.total_comp_energy = 0.0
        self.total_comm_energy = 0.0

    def federated_learnng_fedprox(self, client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=1,
                                   mu=0.01):
        for round_idx in rane(num_rounds):
            selected_loaders = random.sample(client_loaders, clients_per_round)
            models, sample_counts = [], []
            global_state = self.model.state_dict()

            for loader in selected_loaders:
                client = ClientFedProx(self.model, loader, self.device, global_state, mu=mu)
                models.append(client.train(epochs=epochs_per_client))
                sample_counts.append(len(loader.dataset))

                comm, comp_e, comm_e = calculate_round_metrics(epochs_per_client)
                self.total_comm_mb += comm
                self.total_comp_energy += comp_e
                self.total_comm_energy += comm_e

            self.average_weights(models)  

            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.validation_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.numel()

            acc = 100.0 * orrect / max(1, total)
            print(
                f"Round {round_idx + 1} | Acc: {acc:.2f}% | Comm: {self.total_comm_mb:.2f}MB | CompEng: {self.total_comp_energy:.2f}J | CommEng: {self.total_comm_energy:.2f}J")
            with open("test_accuracy_fedprox.txt", mode="a") as f:
                f.write(
                    f"Round {round_idx + 1} | Acc: {acc:.4f}% | Comm(MB): {self.total_comm_mb:.2f} | Comp(J): {self.total_comp_energy:.2f} | CommEng(J): {self.total_comm_energy:.2f}\n")


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data/UCI-HAR Dataset"
    train_dataset = UCIHARDataset(os.path.join(data_dir, "train", "X_train.txt"),
                                  os.path.join(data_dir, "train", "y_train.txt"))
    test_dataset = UCIHARDatset(os.path.join(data_dir, "test", "X_test.txt"),
                                 os.path.join(data_dir, "test", "y_test.txt"))
    validation_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(train_dataset, num_clients=50, batch_size=32,
                                                               dirichlet_alpha=5, seed=42)
    global_model = UCIHARMdel(input_dim=561, hidden_dims=[256, 128, 64], number_of_classes=6, p=0.2)
    server = ServerFedProx(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting Fe===================================")
    server.federated_learnin_fedprox(client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=1, mu=0.01)


if __name__ == "__main__":
    main()
