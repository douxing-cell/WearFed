# motionsense_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MotionSenseDataset(Dataset):
    """
    MotionSense数据集加载器
    将时间序列分割成固定长度窗口

    ✅ 为了与 UCI 版本的 Dirichlet non-IID splitter 完全一致：
      - 保留 self.labels
      - 新增 self.y = torch.LongTensor(labels)
    """
    def __init__(self, data_dir, window_size=128, step_size=64, train=True, transform=None, scaler=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.train = train

        self.activity_map = {
            'dws': 0,
            'jog': 1,
            'sit': 2,
            'std': 3,
            'ups': 4,
            'wlk': 5
        }

        self.feature_columns = [
            'attitude.roll', 'attitude.pitch', 'attitude.yaw',
            'gravity.x', 'gravity.y', 'gravity.z',
            'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
            'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'
        ]

        self.samples = []
        self.labels = []
        self._load_data(train)

        if scaler is None:
            self.scaler = StandardScaler()
            self._normalize_data()
        else:
            self.scaler = scaler
            self._normalize_data()

        # ✅ compatibility: dataset.y for Dirichlet splitter
        self.y = torch.LongTensor(np.asarray(self.labels, dtype=np.int64))

    def _load_data(self, train):
        data_path = os.path.join(self.data_dir, 'A_DeviceMotion_data', 'A_DeviceMotion_data')

        activity_folders = [f for f in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path, f)) and not f.startswith('.')]

        all_files = []
        for activity_folder in activity_folders:
            activity_name = activity_folder.split('_')[0]
            if activity_name not in self.activity_map:
                continue

            activity_path = os.path.join(data_path, activity_folder)
            csv_files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]

            for csv_file in csv_files:
                file_path = os.path.join(activity_path, csv_file)
                all_files.append((file_path, activity_name))

        np.random.seed(42)
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * 0.8)

        files = all_files[:split_idx] if train else all_files[split_idx:]

        for file_path, activity_name in files:
            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in self.feature_columns):
                    continue

                features = df[self.feature_columns].values

                for i in range(0, len(features) - self.window_size + 1, self.step_size):
                    window = features[i:i + self.window_size]
                    if len(window) == self.window_size:
                        self.samples.append(window)
                        self.labels.append(self.activity_map[activity_name])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"Loaded {len(self.samples)} samples for {'train' if train else 'validation'}")

    def _normalize_data(self):
        if len(self.samples) == 0:
            return

        all_data = np.array(self.samples).reshape(-1, len(self.feature_columns))

        if self.train:
            self.scaler.fit(all_data)

        normalized_samples = []
        for sample in self.samples:
            normalized = self.scaler.transform(sample)
            normalized_samples.append(normalized)

        self.samples = normalized_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        sample = torch.FloatTensor(sample)              # (window_size, 12)
        sample = sample.transpose(0, 1).contiguous()    # (12, window_size)

        label = torch.LongTensor([label])[0]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
