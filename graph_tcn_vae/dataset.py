import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, window_size=48, target_dim=10, aux_dim=5, mode='train'):
        self.num_samples = num_samples
        self.window_size = window_size
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.mode = mode
        
        # Generate synthetic data
        # Target: Sine waves + noise
        t = np.linspace(0, 100, window_size)
        self.data_x = []
        self.data_c = []
        self.data_m = []
        
        for _ in range(num_samples):
            # Random phases and frequencies
            freq = np.random.uniform(0.1, 0.5, target_dim)
            phase = np.random.uniform(0, 2*np.pi, target_dim)
            
            x_sample = np.stack([np.sin(freq[i] * t + phase[i]) for i in range(target_dim)], axis=1)
            x_sample = (x_sample + 1) / 2 # Normalize to [0, 1] for Sigmoid
            
            # Aux: Random noise + time features
            c_sample = np.random.rand(window_size, aux_dim)
            
            # Mask: Randomly missing chunks (simulate sensor failure)
            m_sample = np.ones((window_size, target_dim))
            if np.random.rand() > 0.5:
                # Drop a random chunk
                start = np.random.randint(0, window_size // 2)
                length = np.random.randint(5, window_size // 2)
                m_sample[start:start+length, :] = 0
            
            self.data_x.append(x_sample.astype(np.float32))
            self.data_c.append(c_sample.astype(np.float32))
            self.data_m.append(m_sample.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data_x[idx]
        c = self.data_c[idx]
        mask = self.data_m[idx]
        
        # Denoising Strategy (Training only)
        input_x = x.copy()
        input_mask = mask.copy()
        
        if self.mode == 'train':
            # Randomly drop 15% of observed values
            # We only drop where mask is already 1
            prob_drop = 0.15
            drop_indices = np.where((mask == 1) & (np.random.rand(*mask.shape) < prob_drop))
            
            input_mask[drop_indices] = 0
            input_x[drop_indices] = 0 # Zero out missing values
        else:
            # For validation/inference, input is same as ground truth (masked)
            # Or if strictly inference, mask indicates what's missing.
            input_x = x * mask # Ensure missing values are 0
            
        return {
            'target': torch.from_numpy(x),          # Ground Truth
            'condition': torch.from_numpy(c),       # Aux Condition
            'target_mask': torch.from_numpy(mask),  # Ground Truth Mask
            'input_x': torch.from_numpy(input_x),   # Corrupted Input
            'input_mask': torch.from_numpy(input_mask) # Corrupted Mask
        }
