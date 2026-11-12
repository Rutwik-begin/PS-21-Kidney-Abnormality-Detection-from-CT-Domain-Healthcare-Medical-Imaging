"""
loss.py - Specialized Loss Functions for Handling Class Imbalance in Medical CT.
Supports:
1. Standard CrossEntropyLoss
2. Class-Weighted CrossEntropyLoss (penalizing minority class errors, e.g. Stone)
3. Multi-Class Focal Loss (focusing on hard borderline lesions and rare stones)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Down-weights easy background/normal slices, focusing gradient on hard-to-detect
    small kidney stones and ambiguous soft-tissue cyst/tumour margins.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, num_classes) - raw logits
        targets: (batch_size,) - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of true class
        
        focal_term = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(class_counts: List[int], power: float = 0.5) -> torch.Tensor:
    """
    Computes inverse frequency or smoothed inverse frequency class weights:
    w_c = (N / (K * N_c))^power
    Ensures minority classes like Stone receive proportionally higher penalty.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    k = len(counts)
    weights = total / (k * counts)
    if power != 1.0:
        weights = weights ** power
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights


def get_loss_criterion(
    loss_name: str = 'focal',
    class_counts: Optional[List[int]] = None,
    gamma: float = 2.0,
    device: str = 'cpu'
) -> nn.Module:
    """
    Factory to return requested loss function.
    """
    weights = None
    if class_counts is not None:
        weights = compute_class_weights(class_counts).to(device)

    if loss_name.lower() == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name.lower() == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name.lower() == 'focal':
        return FocalLoss(alpha=weights, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

