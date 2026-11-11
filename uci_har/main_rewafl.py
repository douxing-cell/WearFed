# main_rewafl.py
import os
import copy
import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from model import UCIHARModel
from server import Server
from uci_har_dataset import UCIHARDataset


# =====================================================================
# 1. REWAFL 设备状态追踪器
# =====================================================================
class REWAFLClientState:
    def __init__(self, cid: int, num_samples: int):
        self.cid = cid
        self.num_samples = num_samples
        self.battery_J = random.uniform(10000, 30000)
        self.battery_threshold = self.battery_J * 0.1
        self.tx_rate_mbps = random.uniform(1.0, 80.0)
        self.H = 1
        self.u = 0
        self.last_loss = 2.0
        self.last_loss_diff = 0.1
        self.last_comp_energy = 0.1

    def estimate_cost(self):
        # 与其他脚本对齐参数，FULL 约为 0.75 MB
        size_mb = 0.75
        comp_time = self.H * 0.5
        comp_energy = self.H * 5.0

        # 速率是 Mbps，除以8转为 MB/s
        rate_MBps = self.tx_rate_mbps / 8.0
        comm_time_down = size_mb / rate_MBps
        comm_time_up = size_mb / rate_MBps

        # 下发 1W， 上传 2W
        comm_energy = (comm_time_down * 1.0) + (comm_time_up * 2.0)
        comm_time_total = comm_time_down + comm_time_up
        comm_mb_total = size_mb * 2.0

        return comp_time + comm_time_total, comp_energy + comm_energy, comm_mb_total, comp_energy, comm_energy


# =====================================================================
# 2. REWAFL 服务端
# =====================================================================
class ServerREWAFL(Server):
    def __init__(self, model, validation_loader, mps_device):
        super().__init__(model, validation_loader, mps_device)
        self.total_comm_mb = 0.0
        self.total_comp_energy = 0.0
        self.total_comm_energy = 0.0

    def federated_learning_rewafl(self, client_loaders, num_rounds=50, clients_per_round=10):
        num_clients = len(client_loaders)
        clients_state = [REWAFLClientState(i, len(client_loaders[i].dataset)) for i in range(num_clients)]

        # 1. 调高阈值到 200.0，适应真实的几万焦耳的电量基数
        alpha, beta, epsilon_th, delta_H = 1.0, 1.0, 200.0, 1

        for round_idx in range(num_rounds):
            valid_clients = []
            T_target = 5.0

            for c in clients_state:
                # 2. 模拟真实的信道波动 (5 Mbps ~ 80 Mbps)
                c.tx_rate_mbps = random.uniform(5.0, 80.0)

                eps_hat = (c.last_loss_diff * (c.battery_J - c.battery_threshold)) / max(c.last_comp_energy, 0.001)

                # 3. 真正的动态切换逻辑
                if eps_hat > epsilon_th:
                    # 使用 15.0 作为基准因子
                    psi_s = 15.0 / max(c.tx_rate_mbps, 1.0)
                    # 核心修改：使用 round(四舍五入) 代替 ceil(向上取整)
                    # 只有当网络 < 30 Mbps 时，psi_s 才会 >= 0.5，H 才会变成 2
                    c.H = min(2, 1 + round(psi_s * delta_H))
                else:
                    c.H = 1

                t_ir, e_total_ir, comm_mb, comp_e, comm_e = c.estimate_cost()
                stat_util = (c.num_samples * c.H) * c.last_loss
                lat_util = (T_target / t_ir) ** alpha if t_ir > T_target else 1.0

                if e_total_ir < (c.battery_J - c.battery_threshold):
                    eng_util = ((c.battery_J - c.battery_threshold) / e_total_ir) ** beta
                else:
                    eng_util = 0.0

                final_utility = stat_util * lat_util * eng_util
                if eng_util > 0:
                    valid_clients.append((c, final_utility, t_ir, e_total_ir, comm_mb, comp_e, comm_e))


            valid_clients.sort(key=lambda x: x[1], reverse=True)
            selected = valid_clients[:clients_per_round]
            selected_cids = set([x[0].cid for x in selected])

            models, sample_counts = [], []
            for (c_state, util, t_ir, e_total_ir, comm_mb, comp_e, comm_e) in selected:
                loader = client_loaders[c_state.cid]
                local_model, final_loss = self._train_local_rewafl(loader, c_state.H)
                models.append(local_model)
                sample_counts.append(c_state.num_samples)

                c_state.battery_J -= e_total_ir
                c_state.u = 0
                c_state.last_loss_diff = abs(c_state.last_loss - final_loss)
                c_state.last_loss = final_loss
                c_state.last_comp_energy = c_state.H * 5.0

                self.total_comm_mb += comm_mb
                self.total_comp_energy += comp_e
                self.total_comm_energy += comm_e

            for c in clients_state:
                if c.cid not in selected_cids:
                    c.u += 1

            if len(models) > 0:
                self.average_weights(models)  # 简单调用基类硬聚合

            acc = self._evaluate_rewafl()
            print(
                f"Round {round_idx + 1} | Acc: {acc:.2f}% | Comm: {self.total_comm_mb:.2f}MB | CompEng: {self.total_comp_energy:.2f}J | CommEng: {self.total_comm_energy:.2f}J")
            with open("test_accuracy_rewafl.txt", mode="a") as f:
                f.write(
                    f"Round {round_idx + 1} | Acc: {acc:.4f}% | Comm(MB): {self.total_comm_mb:.2f} | Comp(J): {self.total_comp_energy:.2f} | CommEng(J): {self.total_comm_energy:.2f}\n")

    def _train_local_rewafl(self, dataloader, epochs):
        model_copy = copy.deepcopy(self.model).to(self.device)
        model_copy.train()
        optimizer = optim.SGD(model_copy.parameters(), lr=0.1, momentum=0.2)
        criterion = nn.CrossEntropyLoss()
        last_loss = 0.0
        for _ in range(int(epochs)):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model_copy(data.to(self.device))
                loss = criterion(output, target.to(self.device))
                loss.backward()
                optimizer.step()
                last_loss = loss.item()
        return model_copy, last_loss

    def _evaluate_rewafl(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                pred = self.model(data.to(self.device)).argmax(dim=1)
                correct += (pred == target.to(self.device)).sum().item()
                total += target.numel()
        return 100.0 * correct / max(1, total)


def create_federated_loaders_dirichlet_noniid(dataset, num_clients=50, batch_size=32, dirichlet_alpha=1.0, seed=42):
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
    server = ServerREWAFL(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting REWAFL Baseline Training on UCI (IEEE TMC 2024)")
    print("==================================================")
    server.federated_learning_rewafl(client_loaders, num_rounds=50, clients_per_round=10)

    
if __name__ == "__main__":
    main()