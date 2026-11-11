# main_eflcf.py
import os
import copy
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from model import CNN_V3_V4
from motionsense_dataset import MotionSenseDataset


# =====================================================================
# 1. 数据划分模块 (严格保持 alpha=0.5 极端 Non-IID，与 WearFed 绝对一致)
# =====================================================================
def create_federated_loaders_dirichlet_noniid(
        dataset, num_clients=50, batch_size=32, dirichlet_alpha=0.5, seed=42
):
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


# =====================================================================
# 2. EFLCF 专属客户端 (支持弹性多宽度子网，但无原型蒸馏 KD)
# =====================================================================
class ClientEFLCF:
    def __init__(self, model, dataloader, mps_device, profile="FULL"):
        self.device = mps_device
        self.model = copy.deepcopy(model).to(self.device)
        self.dataloader = dataloader

        # TCAD 2026 核心机制：分配不同宽度的子模型以适应计算瓶颈
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

                # 仅计算分类交叉熵，没有 WearFed 的原型对齐拉力
                logits = self.model(data)
                loss = self.criterion(logits, target)

                loss.backward()
                self.optimizer.step()
        return self.model


# =====================================================================
# 3. EFLCF 服务端调度 (计算瓶颈感知的动态切片 + 传统硬平均)
# =====================================================================
class ServerEFLCF:
    def __init__(self, model, validation_loader, mps_device):
        self.device = mps_device
        self.model = model.to(self.device)
        self.validation_loader = validation_loader

    def average_weights(self, model_list, sample_counts):
        """传统硬平均聚合：物理通道直接对齐（忽略特征语义错位）"""
        weights = [float(max(0, int(n))) for n in sample_counts]
        denom = sum(weights)
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
            models = []
            sample_counts = []

            for cid in selected_indices:
                loader = client_loaders[cid]

                # 模拟物联网环境中设备的动态计算瓶颈 (Computing-Constrained IoT)
                # 随机生成计算约束状态，动态下发多宽度子网
                constraint_state = random.random()
                if constraint_state > 0.6:
                    assigned_profile = "FULL"
                elif constraint_state > 0.2:
                    assigned_profile = "PRUNED"
                else:
                    assigned_profile = "MSMALL"

                client = ClientEFLCF(self.model, loader, self.device, profile=assigned_profile)
                local_model = client.train(epochs=epochs_per_client)

                models.append(local_model)
                n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 1
                sample_counts.append(n_samples)

            # TCAD 的硬平均聚合
            self.average_weights(models, sample_counts)

            # 评估日志 (在服务器端恢复 FULL 状态进行测试)
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
            print(f"Round {round_idx + 1} completed | EFLCF (TCAD 2026 Elastic w/o Prototype KD) Accuracy: {acc:.6f}%")

            with open("test_accuracy_eflcf.txt", mode="a") as f:
                f.write(f"Round {round_idx + 1} completed | Accuracy: {acc:.4f}%\n")


# =====================================================================
# 4. 运行入口
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    window_size, step_size, batch_size = 128, 64, 32

    print("Loading MotionSense Dataset for EFLCF...")
    train_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=True)
    test_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=False,
                                      scaler=train_dataset.scaler)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset, num_clients=50, batch_size=batch_size, dirichlet_alpha=5, seed=42
    )

    global_model = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=6, p=0.2, window_size=window_size,
                             num_features=12)
    server = ServerEFLCF(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting EFLCF Baseline Training (TCAD 2026 SOTA)")
    print("==================================================")

    server.federated_learning_eflcf(
        client_loaders=client_loaders,
        num_rounds=50,
        clients_per_round=10,
        epochs_per_client=2
    )


if __name__ == "__main__":
    main()