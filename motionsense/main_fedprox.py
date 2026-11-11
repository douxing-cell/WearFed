# main_fedprox.py
import os
import copy
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from typing import Optional

from model import CNN_V3_V4
from server import Server
from motionsense_dataset import MotionSenseDataset


# =====================================================================
# 1. 核心创新：FedProx 的本地训练客户端 (带有 Proximal Term)
# =====================================================================
class ClientFedProx:
    def __init__(self, model, dataloader, mps_device, global_state, mu=0.01):
        self.device = mps_device
        # 客户端拉取全局模型并复制到本地
        self.model = copy.deepcopy(model).to(self.device)
        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.2)
        self.criterion = nn.CrossEntropyLoss()

        # 冻结并保存全局模型的权重，用于计算近端惩罚项
        self.global_state = {k: v.clone().detach().to(self.device) for k, v in global_state.items()}
        self.mu = mu  # 惩罚系数 μ

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)

                # 1. 计算正常的交叉熵损失
                logits = self.model(data)
                loss_ce = self.criterion(logits, target)

                # 2. 计算 FedProx 的核心：近端项 (Proximal Term) => (μ/2) * ||w - w_t||^2
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.global_state:
                        proximal_term += ((param - self.global_state[name]) ** 2).sum()

                # 3. 总损失 = CE + 惩罚
                loss = loss_ce + (self.mu / 2.0) * proximal_term

                loss.backward()
                self.optimizer.step()

        return self.model


# =====================================================================
# 2. FedProx 的服务端调度 (继承并重写你的 Server)
# =====================================================================
class ServerFedProx(Server):
    def federated_learning_fedprox(
            self, client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=1, mu=0.01
    ):
        for round_idx in range(num_rounds):
            selected_loaders = random.sample(client_loaders, clients_per_round)
            models = []
            sample_counts = []

            # 获取本轮最新的全局模型状态
            global_state = self.model.state_dict()

            for loader in selected_loaders:
                # 分发给 FedProx 专属的 Client 进行带有惩罚项的训练
                client = ClientFedProx(self.model, loader, self.device, global_state, mu=mu)
                local_model = client.train(epochs=epochs_per_client)

                models.append(local_model)
                n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 1
                sample_counts.append(n_samples)

            # 调用你写好的按样本量加权聚合函数
            self.average_weights(models, sample_counts)

            # 评估并保存专门的 FedProx 日志
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.validation_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.numel()

            acc = 100.0 * correct / max(1, total)
            print(f"Round {round_idx + 1} completed | FedProx Validation Accuracy: {acc:.6f}%")

            # 写入专门的 txt 文件防覆盖
            with open("test_accuracy_fedprox.txt", mode="a") as f:
                f.write(f"Round {round_idx + 1} completed\n")
                f.write(f"Validation Accuracy: {acc:.6f}%\n")


# =====================================================================
# 3. 数据划分逻辑 (保持与 WearFed 绝对一致)
# =====================================================================
def create_federated_loaders_dirichlet_noniid(dataset, num_clients=20, batch_size=32, dirichlet_alpha=1,
                                              min_samples_per_client=10, seed=42):
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
        if min([len(ci) for ci in client_indices]) >= min_samples_per_client:
            break

    loaders = []
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])
        loaders.append(DataLoader(Subset(dataset, client_indices[cid]), batch_size=batch_size, shuffle=True))
    return loaders


# =====================================================================
# 4. 运行入口
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"  # MotionSense 路径
    window_size, step_size, batch_size = 128, 64, 32

    print("Loading MotionSense Dataset for FedProx...")
    train_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=True)
    test_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=False,
                                      scaler=train_dataset.scaler)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset, num_clients=50, batch_size=batch_size, dirichlet_alpha=1, seed=24
    )

    global_model = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=6, p=0.2, window_size=window_size,
                             num_features=12)
    server = ServerFedProx(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting FedProx Baseline Training (mu=0.01)")
    print("==================================================")

    server.federated_learning_fedprox(
        client_loaders=client_loaders,
        num_rounds=50,
        clients_per_round=10,
        epochs_per_client=2,
        mu=0.01  # FedProx 的超参数 μ
    )


if __name__ == "__main__":
    main()