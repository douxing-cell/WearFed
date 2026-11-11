
import os
import copy
import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from typing import Optional, List, Dict

from model import CNN_V3_V4
from server import Server





class REWAFLClientState:
    def __init__(self, cid: int, num_samples: int):
        self.cid = cid
        self.num_samples = num_samples

        
        self.battery_J = random.uniform(10000, 30000
        self.battery_threshold = self.battery_J * 0.1  
        self.tx_rate_mbps = random.uniform(1.0, 80.0)

        
        self.H = 1  
        self.u = 0  
        self.last_loss = 2.0 
        self.last_lossdiff = 0.1  
        self.last_comp_energy = 0.1

    def estimate_cost(self):
        
        
        comp_time = self.H * 0.5
        comp_energy = self.H * 5.0

        model_size_mb = 16.0  
        comm_time = model_size_mb / self.tx_rate_mbps
        comm_energy = comm_time * 2.0  

        return comp_time + comm_time, comp_energy + comm_energy





class ServerREWAFL(Server):
    def federated_learning_rewafl(
            self, client_loaders, num_rounds=50, clients_per_round=10
    ):
        num_clients = len(client_loaders)
        
        clients_state = [REWAFLClientState(i, len(client_loaders[i].dataset)) for i in range(num_clients)]

        alpha, beta = 1.0, 1.0  
        epsilon_th = 0.01  
        delta_H = 1  

        for round_idx in range(num_rounds):
            
            utilities = []
            valid_clients = []

            
            T_target = 5.0

            for c in clients_state:
                
                eps_hat = (c.last_loss_diff * (c.battery_J - c.battery_threshold)) / max(c.last_comp_energy, 0.001)

                
                if eps_hat > epsilon_th:
                    psi_s = 10.0 / max(c.tx_rate_mbps, 1.0)  
                    c.H = min(3, c.H + math.ceil(psi_s * delta_H))  

                
                t_ir, e_ir = c.estimate_cost()

                
                
                stat_util = (c.num_samples * c.H) * c.last_loss

                
                lat_util = (T_target / t_ir) ** alpha if t_ir > T_target else 1.0

                
                if e_ir < (c.battery_J - c.battery_threshold):
                    eng_util = ((c.battery_J - c.battery_threshold) / e_ir) ** beta
                else:
                    eng_util = 0.0  

                final_utility = stat_util * lat_util * eng_util

                if eng_util > 0:
                    valid_clients.append((c, final_utility, t_ir, e_ir))

            
            valid_clients.sort(key=lambda x: x[1], reverse=True)
            selected = valid_clients[:clients_per_round]
            selected_cids = set([x[0].cid for x in selected])

            
            models = []
            sample_counts = []

            for (c_state, util, t_ir, e_ir) in selected:
                loader = client_loaders[c_state.cid]
                
                local_model, final_loss = self._train_local_rewafl(loader, c_state.H)
                models.append(local_model)
                sample_counts.append(c_state.num_samples)

                
                c_state.battery_J -= e_ir  
                c_state.u = 0  
                c_state.last_loss_diff = abs(c_state.last_loss - final_loss)
                c_state.last_loss = final_loss
                c_state.last_comp_energy = c_state.H * 5.0

            
            for c in clients_state:
                if c.cid not in selected_cids:
                    c.u += 1

            
            if len(models) > 0:
                self.average_weights(models, sample_counts)

            acc = self._evaluate_rewafl()
            print(f"Round {round_idx + 1} | REWAFL Acc: {acc:.2f}% | Selected: {[x[0].cid for x in selected]}")

            with open("test_accuracy_rewafl.txt", mode="a") as f:
                f.write(f"Round {round_idx + 1} completed | Accuracy: {acc:.4f}%\n")

    def _train_local_rewafl(self, dataloader, epochs):
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()
        optimizer = optim.SGD(model_copy.parameters(), lr=0.1, momentum=0.2)
        criterion = nn.CrossEntropyLoss()

        last_loss = 0.0
        for _ in range(int(epochs)):
            for data, target in dataloader:
                optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = model_copy(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                last_loss = loss.item()
        return model_copy, last_loss

    def _evaluate_rewafl(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
        return 100.0 * correct / max(1, total)





def create_federated_loaders_dirichlet_noniid(dataset, num_clients=50, batch_size=32, alpha=10, seed=42):
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
            proportions = np.random.dirichlet([alpha] * num_clients)
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





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"

    print("Loading MotionSense Dataset for REWAFL...")
    train_dataset = MotionSenseDataset(data_dir=data_dir, window_size=128, step_size=64, train=True)
    test_dataset = MotionSenseDataset(data_dir=data_dir, window_size=128, step_size=64, train=False,
                                      scaler=train_dataset.scaler)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    
    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset, num_clients=50, batch_size=32, alpha=2, seed=42
    )

    global_model = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=6, p=0.2, window_size=128,
                             num_features=12)
    server = ServerREWAFL(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting REWAFL Baseline Training (IEEE TMC 2024)")
    print("==================================================")

    server.federated_learning_rewafl(
        client_loaders=client_loaders,
        num_rounds=50,
        clients_per_round=10
    )


if __name__ == "__main__":
    main()
