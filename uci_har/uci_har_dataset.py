import numpy as np
import torch
from torch.utils.data import Dataset

class UCIHARDataset(Dataset):
    """
    UCI HAR Dataset加载器
    数据集包含561维特征向量和6个活动类别
    """
    def __init__(self, X_file, y_file, transform=None):
        """
        Args:
            X_file: 特征文件路径 (X_train.txt 或 X_test.txt)
            y_file: 标签文件路径 (y_train.txt 或 y_test.txt)
            transform: 可选的变换函数
        """
        # 加载数据
        self.X = np.loadtxt(X_file)
        self.y = np.loadtxt(y_file, dtype=np.int64)
        
        # 标签从1-6转换为0-5（PyTorch需要从0开始）
        self.y = self.y - 1
        
        # 转换为torch tensor
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

