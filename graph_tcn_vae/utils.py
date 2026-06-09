import torch
import numpy as np

class KLAnnealingScheduler:
    def __init__(self, total_epochs, warmup_epochs, min_beta=0.0, max_beta=1.0, strategy='cosine', n_cycles=4, ratio=0.5):
        """
        KL Annealing Scheduler.
        
        For 'cosine' and 'linear' strategies:
            - warmup_epochs controls WHEN beta reaches max_beta
            - β starts at 0 and reaches max_beta at warmup_epochs
            - After warmup_epochs, β stays at max_beta
            
        For 'cyclical' strategy:
            - Ignores warmup_epochs, uses n_cycles and ratio instead
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.strategy = strategy
        self.n_cycles = n_cycles
        self.ratio = ratio

    def get_beta(self, epoch):
        if self.strategy == 'cyclical':
            # Cyclical Annealing
            # Ignores global warmup_epochs in favor of cyclic behavior
            period = self.total_epochs / self.n_cycles
            step_in_cycle = epoch % period
            cycle_progress = step_in_cycle / period
            
            if cycle_progress < self.ratio:
                # Linear increase
                beta = (cycle_progress / self.ratio) * self.max_beta
            else:
                # Plateau
                beta = self.max_beta
            return beta

        # For linear and cosine: warmup_epochs is when we REACH max
        if epoch >= self.warmup_epochs:
            return self.max_beta
        
        progress = epoch / self.warmup_epochs  # 0 → 1 during warmup
        
        if self.strategy == 'linear':
            beta = self.min_beta + progress * (self.max_beta - self.min_beta)
        elif self.strategy == 'cosine':
            # Cosine from 0 to max at warmup_epochs
            beta = self.min_beta + 0.5 * (self.max_beta - self.min_beta) * (1 - np.cos(progress * np.pi))
        else:
            beta = self.max_beta
            
        return min(max(beta, self.min_beta), self.max_beta)

class LRWarmupCosineScheduler:
    """
    Learning Rate Scheduler with Linear Warmup + Cosine Annealing.
    
    - Linear warmup: LR increases from 0 to max_lr during warmup_epochs
    - Cosine annealing: LR decreases from max_lr to min_lr after warmup
    """
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.max_lr = optimizer.param_groups[0]['lr']
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 -> max_lr
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: max_lr -> min_lr
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
