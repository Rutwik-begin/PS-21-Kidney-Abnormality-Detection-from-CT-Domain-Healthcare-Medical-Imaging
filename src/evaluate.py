"""
evaluate.py - Evaluation Metrics, Confusion Matrix, and ROC Analysis for Medical CT.
Highlights:
- Test Accuracy
- Macro F1 Score
- Explicit Stone Recall (smallest class requirement)
- Per-Class Precision, Recall, and F1
- Multi-Class ROC-AUC
- High-resolution visual artifacts for report and documentation
"""

import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumour']

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference over dataloader and returns true labels, predicted labels, and probability distributions.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict:
    """
    Calculates key evaluation metrics with special focus on Stone Recall.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * sup) / np.sum(sup)

    # Stone index
    stone_idx = class_names.index('Stone')
    stone_recall = rec[stone_idx]
    stone_precision = prec[stone_idx]
    stone_f1 = f1[stone_idx]

    # ROC AUC
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = float('nan')

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1_score': float(f1[i]),
            'support': int(sup[i])
        }

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_auc': float(macro_auc),
        'stone_recall': float(stone_recall),
        'stone_precision': float(stone_precision),
        'stone_f1': float(stone_f1),
        'per_class': per_class
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
    title: str = "Kidney Abnormality Detection Confusion Matrix"
):
    """
    Generates a clean normalized confusion matrix heatmap suitable for medical publications.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    plt.title(title, fontsize=12, pad=12, fontweight='bold')
    plt.xlabel('Predicted Medical Diagnosis', fontsize=11, fontweight='bold')
    plt.ylabel('True Ground Truth Diagnosis', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None
):
    """
    Plots One-vs-Rest ROC curves for each pathology.
    """
    plt.figure(figsize=(7, 6), dpi=300)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    plt.title('Multi-Class ROC Curves by Renal Abnormality', fontsize=12, pad=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
