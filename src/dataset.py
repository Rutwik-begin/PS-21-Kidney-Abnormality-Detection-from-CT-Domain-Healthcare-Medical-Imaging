"""
dataset.py - Dataset loading, preprocessing, patient-level grouping, and augmentation
for PS-21 Kidney Abnormality Detection from CT.
"""

import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Class mapping
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumour']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Class alias normalization (handling 'Tumor' vs 'Tumour')
ALIAS_MAP = {
    'cyst': 'Cyst',
    'normal': 'Normal',
    'stone': 'Stone',
    'tumor': 'Tumour',
    'tumour': 'Tumour'
}

def extract_patient_id(filename: str, slices_per_study: int = 40) -> str:
    """
    Extracts or approximates patient study ID from image filename.
    In the Kaggle CT Kidney dataset, filenames are sequential cuts
    (e.g., 'Cyst- (105).jpg', 'Stone- (42).jpg').
    Consecutive slices correspond to axial passes through patient volumetric scans.
    """
    stem = Path(filename).stem
    match = re.search(r'\((\d+)\)', stem)
    if match:
        slice_idx = int(match.group(1))
        # Group contiguous slices into distinct patient study IDs
        patient_num = (slice_idx - 1) // slices_per_study + 1
        prefix = stem.split('-')[0].strip()
        return f"{prefix}_patient_{patient_num:03d}"
    
    # Fallback to prefix
    return f"unknown_patient_{stem}"


class KidneyCTDataset(Dataset):
    """
    PyTorch Dataset for Kidney CT Slices.
    Supports on-the-fly transform application and metadata extraction.
    """
    def __init__(self, file_paths: List[str], labels: List[int], patient_ids: List[str], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        return image, label, patient_id


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and validation/test transforms.
    CT imaging specific:
    - Bilateral horizontal flip is clinically valid (left/right kidney symmetry).
    - Avoid arbitrary vertical flip to preserve standard anterior/posterior CT orientation.
    - Subtle rotation (+/- 10 deg) simulates patient tilt in gantry.
    - Mild contrast adjustment simulates soft tissue window variations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def scan_dataset(root_dir: str) -> pd.DataFrame:
    """
    Scans the dataset directory, finds all images, maps labels and patient IDs.
    """
    root_path = Path(root_dir)
    records = []

    for path in root_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            parent_name = path.parent.name.lower()
            stem_prefix = path.name.split('-')[0].strip().lower()

            # Identify class
            matched_class = None
            for key, cname in ALIAS_MAP.items():
                if key in parent_name or key == stem_prefix:
                    matched_class = cname
                    break

            if matched_class:
                patient_id = extract_patient_id(path.name)
                records.append({
                    'file_path': str(path.resolve()),
                    'filename': path.name,
                    'class_name': matched_class,
                    'label': CLASS_TO_IDX[matched_class],
                    'patient_id': patient_id
                })

    df = pd.DataFrame(records)
    return df


def create_dataloaders(
    df: pd.DataFrame,
    split_strategy: str = 'patient_grouped', # 'patient_grouped' vs 'random_leakage'
    batch_size: int = 32,
    num_workers: int = 2,
    val_size: float = 0.15,
    test_size: float = 0.15,
    img_size: int = 224,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Constructs Train, Validation, and Test dataloaders under either:
    1. 'patient_grouped': No slice from the same patient appears in multiple splits (Prevents Leakage).
    2. 'random_leakage': Naive slice-level random split (Causes Information Leakage).
    """
    train_transform, test_transform = get_transforms(img_size)

    if split_strategy == 'patient_grouped':
        # Split by patient groups to prevent data leakage
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df['patient_id']))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split remaining into train and validation by patient
        adjusted_val_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['patient_id']))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    elif split_strategy == 'random_leakage':
        # Naive random stratified split (causes leakage across patient studies)
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=random_state
        )
        adjusted_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=adjusted_val_size, stratify=train_val_df['label'], random_state=random_state
        )
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    train_ds = KidneyCTDataset(train_df['file_path'].tolist(), train_df['label'].tolist(), train_df['patient_id'].tolist(), transform=train_transform)
    val_ds = KidneyCTDataset(val_df['file_path'].tolist(), val_df['label'].tolist(), val_df['patient_id'].tolist(), transform=test_transform)
    test_ds = KidneyCTDataset(test_df['file_path'].tolist(), test_df['label'].tolist(), test_df['patient_id'].tolist(), transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    metadata = {
        'strategy': split_strategy,
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

    return train_loader, val_loader, test_loader, metadata
