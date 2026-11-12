"""
train.py - Model Training Engine with Early Stopping and Metric Tracking.
Supports:
- Mixed precision training (AMP)
- Early stopping based on Validation Macro F1
- Real-time logging of Loss, Accuracy, Macro F1, and Stone Recall
- Learning rate scheduling (Cosine Annealing / Reduce on Plateau)
"""

import os
import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumour']

class EarlyStopping:
    """
    Early stops training when validation metric stops improving.
    """
    def __init__(self, patience: int = 5, mode: str = 'max', delta: float = 1e-4, save_path: str = 'best_model.pth'):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        improved = (current_score > self.best_score + self.delta) if self.mode == 'max' else (current_score < self.best_score - self.delta)
        
        if improved:
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, model: nn.Module):
        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Trains model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.startswith('cuda'):
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    class_names=CLASS_NAMES
) -> Tuple[float, float, float, float]:
    """
    Evaluates model on validation/test set.
    Returns: (loss, accuracy, macro_f1, stone_recall)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    _, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, labels=range(len(class_names)), zero_division=0)
    
    macro_f1 = float(np.mean(f1))
    stone_idx = class_names.index('Stone')
    stone_recall = float(rec[stone_idx])

    return epoch_loss, epoch_acc, macro_f1, stone_recall


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 15,
    device: str = 'cuda',
    early_stopping_patience: int = 4,
    checkpoint_path: str = 'checkpoints/best_model.pth'
) -> Dict:
    """
    Main training loop managing epochs, scheduling, early stopping, and history tracking.
    """
    model = model.to(device)
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode='max', save_path=checkpoint_path)
    scaler = torch.amp.GradScaler('cuda') if device.startswith('cuda') else None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_f1': [],
        'val_stone_recall': [],
        'lr': [],
        'training_time_sec': 0.0
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_macro_f1, val_stone_rec = evaluate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)
        history['val_stone_recall'].append(val_stone_rec)
        history['lr'].append(current_lr)

        ep_dur = time.time() - ep_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_dur:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}, Stone-Rec: {val_stone_rec:.4f} | "
              f"LR: {current_lr:.2e}")

        # Early stopping check on validation Macro F1
        improved = early_stopper(val_macro_f1, model)
        if improved:
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Val Macro-F1: {val_macro_f1:.4f})")
        if early_stopper.early_stop:
            print(f"  [!] Early stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
            break

    total_time = time.time() - start_time
    history['training_time_sec'] = total_time
    print(f"\n[+] Training completed in {total_time:.2f} seconds ({total_time/60:.2f} min).")

    # Load best weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Loaded best model checkpoint from {checkpoint_path}")

    return history

