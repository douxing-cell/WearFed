# main_dyn_prune.py
import os
import copy
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from model import CNN_V3_V4


# =====================================================================
# 1. 数据划分模块
# =====================================================================
def create_federated_loaders_dirichlet_noniid(
        dataset, num_clients=50, batch_size=32, dirichlet_alpha=5.0, seed=42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    y = dataset.y.cpu().numpy()
    num_classes = it(y.max() + 1
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
# 2. 动态剪枝客户端 (被动接收全局动态策略)
# =====================================================================
class ClientDynPrune:
    def __init__(self, mdel, dataloader, mps_device, current_profile):
        self.device = mps_device
        self.model = copy.deepcopy(model).to(self.device)
        self.dataloader = dataloader

        # 接收服务端在当前轮次动态下发的统一切片配置
        if hasattr(self.model, "set_profile"):
            self.model.set_profile(current_profile)

        # 采用温和的学习率并保留安全带，保证这个强力基线不会崩溃
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        self.model.train()
        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)

                logits = self.model(data)
                loss = self.criterion(logits, target)
                loss.backward()

                # 防梯度爆炸安全带
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                self.optimizer.step()

        return self.model


# =====================================================================
# 3. 动态剪枝服务端 (全局自适应压缩调度)
# =====================================================================
class ServerDynPrune:
    def __init__(self, model, validation_loader, mps_device):
        self.device = mps_device
        self.model = model.to(self.device)
        self.validation_loader = validation_loader

    def average_weights(self, model_list, sample_counts):
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

    def federated_learning_dyn_prune(self, client_loaders, num_rounds=50, clients_per_round=10, epochs_per_client=2):
        for round_idx in range(num_rounds):

            # 🌟 核心动态剪枝调度逻辑 (随训练轮次全局自适应压缩)
            # 初期使用大模型提取特征，中后期动态缩小模型以降低全局能耗
            if round_idx < 15:
                current_profile = "FULL"
            elif round_idx < 35:
                current_profile = "PRUNED"
            else:
                current_profile = "MSMALL"

            # 保持服务端模型与下发配置一致
            if hasattr(self.model, "set_profile"):
                self.model.set_profile(current_profile)

            selected_indices = random.sample(range(len(client_loaders)), clients_per_round)
            models = []
            sample_counts = []

            for cid in selected_indices:
                loader = client_loaders[cid]
                # 所有被选中的设备强制采用统一的动态配置，无视底层个体电量
                client = ClientDynPrune(self.model, loader, self.device, current_profile=current_profile)
                local_model = client.train(epochs=epochs_per_client)

                models.append(local_model)
                n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 1
                sample_counts.append(n_samples)

            # 参数聚合
            self.average_weights(models, sample_counts)

            # 恢复到 FULL 状态进行一致性评估
            self.model.eval()
            # if hasattr(self.model, "set_profile"):
            #     self.model.set_profile("FULL")

            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.validation_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.numel()

            acc = 100.0 * correct / max(1, total)
            print(f"Round {round_idx + 1} | Profile: {current_profile} | Dyn-Prune Acc: {acc:.4f}%")

            with open("test_accuracy_dyn_prune.txt", mode="a") as f:
                f.write(f"Round {round_idx + 1} completed | Profile: {current_profile} | Accuracy: {acc:.4f}%\n")


# =====================================================================
# 4. 运行入口
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    window_size, step_size, batch_size = 128, 64, 32

    print("Loading MotionSense Dataset for Dyn-Prune Baseline...")
    train_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=True)
    test_dataset = MotionSenseDataset(data_dir=data_dir, window_size=window_size, step_size=step_size, train=False,
                                      scaler=train_dataset.scaler)
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    client_loaders = create_federated_loaders_dirichlet_noniid(
        train_dataset, num_clients=50, batch_size=batch_size, dirichlet_alpha=5, seed=42
    )

    global_model = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=6, p=0.2, window_size=window_size,
                             num_features=12)
    server = ServerDynPrune(global_model, validation_loader, device)

    print("\n==================================================")
    print("🚀 Starting Dynamic Pruning Baseline Training")
    print("==================================================")

    server.federated_learning_dyn_prune(
        client_loaders=client_loaders,
        num_rounds=50,
        clients_per_round=10,
        epochs_per_client=3
    )


if __name__ == "__main__":
    main()
