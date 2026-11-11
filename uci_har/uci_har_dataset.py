import numpy as np
import torch
from torch.utils.data import Dataset

class UCIHARDataset(Dataset):
    



    def __init__(self, X_file, y_file, transform=None):
        





        
        self.X = np.loadtxt(X_file)
        self.y = np.loadtxt(y_file, dtype=np.int64)
        
        
        self.y = self.y - 1
        
        
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

