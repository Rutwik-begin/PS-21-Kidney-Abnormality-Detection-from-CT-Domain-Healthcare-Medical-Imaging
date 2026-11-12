"""
generate_sample_format_figures.py - Generates exact replica figures matching the user's reference report style.
1. sample_fig1_class_dist.png (Page 1 bar chart)
2. sample_fig2_left_curves.png (Page 2 left learning curves)
3. sample_fig2_right_cm.png (Page 2 right confusion matrix)
4. sample_fig2_combined.png (Side-by-side combined figure)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

FIG_DIR = Path("D:/CBL/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Page 1 Fig 1: Class distribution bar chart matching reference style
def make_fig1():
    classes = ['cyst', 'normal', 'stone', 'tumor']
    counts = [3709, 5077, 1377, 2283]

    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=300)
    
    # Simple clean bars matching the reference
    bars = ax.bar(classes, counts, color='#1f77b4', width=0.55, edgecolor='none')

    # Add counts above bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{count:,}",
            ha='center', va='bottom', fontsize=8
        )

    ax.set_title('CT Kidney Dataset class distribution (12,446 images)', fontsize=9, pad=6)
    ax.set_ylabel('Images', fontsize=8)
    ax.set_ylim(0, 5800)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Box frame matching reference
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sample_fig1_class_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated sample_fig1_class_dist.png")

# 2. Page 2 Fig 2 Left: Synthetic learning curves
def make_fig2_curves():
    epochs = np.arange(1, 16)
    train_loss = 1.6 * np.exp(-epochs / 5.0) + 0.15
    val_loss = 1.4 * np.exp(-epochs / 6.0) + 0.28
    val_macro_f1 = 0.4 + 0.56 / (1.0 + np.exp(-(epochs - 4.5) / 2.2))

    fig, ax1 = plt.subplots(figsize=(3.4, 2.2), dpi=300)

    # Left axis: Loss
    l1 = ax1.plot(epochs, train_loss, color='#1f77b4', lw=1.8, label='Train loss')
    l2 = ax1.plot(epochs, val_loss, color='#ff7f0e', lw=1.8, label='Val. loss')
    ax1.set_xlabel('Epoch', fontsize=8)
    ax1.set_ylabel('Loss', fontsize=8)
    ax1.set_ylim(0, 1.8)
    ax1.tick_params(axis='both', labelsize=7.5)

    # Right axis: Macro F1
    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, val_macro_f1, color='#1f77b4', lw=1.8, linestyle='--', label='Val. macro F1')
    ax2.set_ylabel('Macro F1', fontsize=8)
    ax2.set_ylim(0.3, 1.0)
    ax2.tick_params(axis='both', labelsize=7.5)

    # Combined legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=6.5, frameon=True)

    plt.title('Synthetic learning curves', fontsize=8.5, pad=5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sample_fig2_left_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated sample_fig2_left_curves.png")

# 3. Page 2 Fig 2 Right: Synthetic test confusion matrix
def make_fig2_cm():
    classes = ['cyst', 'normal', 'stone', 'tumor']
    cm = np.array([
        [534,   4,   3,  15],
        [  3, 742,   5,  12],
        [  2,   6, 196,   3],
        [ 14,   8,   4, 316]
    ])

    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={'size': 7.5},
        ax=ax
    )
    ax.set_title('Synthetic test confusion matrix', fontsize=8.5, pad=5)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.tick_params(axis='both', labelsize=7.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sample_fig2_right_cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated sample_fig2_right_cm.png")

# 4. Combined Side-by-Side Figure 2
def make_fig2_combined():
    epochs = np.arange(1, 16)
    train_loss = 1.6 * np.exp(-epochs / 5.0) + 0.15
    val_loss = 1.4 * np.exp(-epochs / 6.0) + 0.28
    val_macro_f1 = 0.4 + 0.56 / (1.0 + np.exp(-(epochs - 4.5) / 2.2))

    classes = ['cyst', 'normal', 'stone', 'tumor']
    cm = np.array([
        [534,   4,   3,  15],
        [  3, 742,   5,  12],
        [  2,   6, 196,   3],
        [ 14,   8,   4, 316]
    ])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=300)

    # Left: Curves
    l1 = ax1.plot(epochs, train_loss, color='#1f77b4', lw=1.6, label='Train loss')
    l2 = ax1.plot(epochs, val_loss, color='#ff7f0e', lw=1.6, label='Val. loss')
    ax1.set_xlabel('Epoch', fontsize=8)
    ax1.set_ylabel('Loss', fontsize=8)
    ax1.set_ylim(0, 1.8)
    ax1.tick_params(axis='both', labelsize=7)

    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, val_macro_f1, color='#1f77b4', lw=1.6, linestyle='--', label='Val. macro F1')
    ax2.set_ylabel('Macro F1', fontsize=8)
    ax2.set_ylim(0.3, 1.0)
    ax2.tick_params(axis='both', labelsize=7)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=6.2, frameon=True)
    ax1.set_title('Synthetic learning curves', fontsize=8.5, pad=5)

    # Right: Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={'size': 7},
        ax=ax3
    )
    ax3.set_title('Synthetic test confusion matrix', fontsize=8.5, pad=5)
    ax3.set_xlabel('Predicted', fontsize=8)
    ax3.set_ylabel('True', fontsize=8)
    ax3.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sample_fig2_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated sample_fig2_combined.png")

if __name__ == '__main__':
    make_fig1()
    make_fig2_curves()
    make_fig2_cm()
    make_fig2_combined()
    print("[+] All reference-styled figures generated successfully!")
