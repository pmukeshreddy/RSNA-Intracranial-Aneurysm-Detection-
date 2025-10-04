import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Tuple, Optional
from tqdm import tqdm

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

class Aneurysm3DDataset(Dataset):
    """
    Dataset for 3D volumes from 2.5D preprocessed stacks.
    Converts (D, 3, H, W) -> (3, 64, 128, 128) for 3D CNNs.
    """
    
    def __init__(
        self, 
        npy_dir: str,
        train_df: pd.DataFrame,
        fold: int,
        is_training: bool = True,
        cache_data: bool = False,
        transform: Optional[callable] = None,
        channel_mode: str = 'average'  # 'average', 'first', or 'max'
    ):
        self.npy_dir = Path(npy_dir)
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        self.is_training = is_training
        self.channel_mode = channel_mode

        print("Building file index...")
        self.file_mapping = {}

        for root, dirs, files in os.walk(self.npy_dir):
            for f in files:
                if f.endswith('.npy'):
                    series_id = f.replace('.npy', '')
                    full_path = Path(root) / f
                    self.file_mapping[series_id] = full_path

        print(f"   Found {len(self.file_mapping)} .npy files")

        if is_training:
            self.df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        else:
            self.df = train_df[train_df['fold'] == fold].reset_index(drop=True)

        available_series = set(self.file_mapping.keys())
        self.df = self.df[self.df['SeriesInstanceUID'].isin(available_series)].reset_index(drop=True)
        
        print(f"   Dataset size: {len(self.df)} samples")

        if cache_data:
            print("Caching data in memory...")
            for idx in tqdm(range(len(self.df)), desc="   Caching"):
                self._load_data(idx)

    def __len__(self) -> int:
        return len(self.df)

    def _load_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        series_id = self.df.iloc[idx]['SeriesInstanceUID']

        if series_id in self.cache:
            return self.cache[series_id]

        npy_path = self.file_mapping.get(series_id)
        if npy_path is None:
            print(f"File not found: {series_id}")
            return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)

        try:
            # Load: shape (D, 3, H, W)
            stacks = np.load(npy_path).astype(np.float32)
            
            if stacks.ndim != 4:
                print(f"Unexpected shape {stacks.shape} for {series_id}, expected (D, 3, H, W)")
                return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)
            
            # Collapse 3 channels per slice into single depth volume
            if self.channel_mode == 'average':
                volume = stacks.mean(axis=1)  # (D, H, W)
            elif self.channel_mode == 'first':
                volume = stacks[:, 0, :, :]  # (D, H, W)
            elif self.channel_mode == 'max':
                volume = stacks.max(axis=1)  # (D, H, W)
            else:
                volume = stacks.mean(axis=1)
            
            # Replicate to 3 channels for model: (3, D, H, W)
            volume = np.stack([volume, volume, volume], axis=0)
            
            # Resize to target: (3, 64, 128, 128)
            volume = self._resize_volume(volume, target_size=(64, 128, 128))
            
            # Get labels
            labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)

            if self.cache_data:
                self.cache[series_id] = (volume, labels)

            return volume, labels

        except Exception as e:
            print(f"Error loading {series_id}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)

    def _resize_volume(self, volume, target_size=(64, 128, 128)):
        """Resize volume to target size (D, H, W)"""
        C, D, H, W = volume.shape
        target_d, target_h, target_w = target_size

        # Depth
        if D != target_d:
            if D > target_d:
                start_idx = (D - target_d) // 2
                volume = volume[:, start_idx:start_idx + target_d, :, :]
            else:
                pad_needed = target_d - D
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                volume = np.pad(volume, ((0, 0), (pad_before, pad_after), (0, 0), (0, 0)),
                              mode='constant', constant_values=0)

        # Height
        C, D, H, W = volume.shape
        if H != target_h:
            if H > target_h:
                start_idx = (H - target_h) // 2
                volume = volume[:, :, start_idx:start_idx + target_h, :]
            else:
                pad_needed = target_h - H
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)),
                              mode='constant', constant_values=0)

        # Width
        C, D, H, W = volume.shape
        if W != target_w:
            if W > target_w:
                start_idx = (W - target_w) // 2
                volume = volume[:, :, :, start_idx:start_idx + target_w]
            else:
                pad_needed = target_w - W
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                volume = np.pad(volume, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)

        return volume

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        volume, labels = self._load_data(idx)

        X = torch.tensor(volume, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32)

        if self.transform and self.is_training:
            X = self.transform(X)

        return X, Y


def custom_collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    X_batch = torch.stack(images, dim=0)
    Y_batch = torch.stack(labels, dim=0)

    return X_batch, Y_batch
