import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Tuple, Optional
from tqdm import tqdm

# Define label columns
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

class Aneurysm2_5DDataset(Dataset):
    """
    Dataset for 2.5D preprocessed DICOM stacks.
    Expects .npy files with shape (num_stacks, channels, H, W)
    Data is already normalized to [0,1] from preprocessing pipeline.
    """
    
    def __init__(
        self, 
        npy_dir: str,
        train_df: pd.DataFrame,
        fold: int,
        is_training: bool = True,
        cache_data: bool = False,
        transform: Optional[callable] = None,
        stack_selection: str = 'middle'  # 'middle', 'random', or 'all'
    ):
        """
        Args:
            npy_dir: Directory containing .npy files
            train_df: DataFrame with SeriesInstanceUID and labels
            fold: Fold number for validation
            is_training: If True, use all folds except `fold`
            cache_data: If True, cache all data in memory
            transform: Optional transform to apply
            stack_selection: How to select from multiple stacks ('middle', 'random', 'all')
        """
        self.npy_dir = Path(npy_dir)
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        self.is_training = is_training
        self.stack_selection = stack_selection

        print("🔍 Building file index...")
        self.file_mapping = {}

        for root, dirs, files in os.walk(self.npy_dir):
            for f in files:
                if f.endswith('.npy'):
                    series_id = f.replace('.npy', '')
                    full_path = Path(root) / f
                    self.file_mapping[series_id] = full_path

        print(f"   Found {len(self.file_mapping)} .npy files")

        # Split data
        if is_training:
            self.df = train_df[train_df['fold'] != fold].reset_index(drop=True)
            print(f"   Training mode: using all folds except fold {fold}")
        else:
            self.df = train_df[train_df['fold'] == fold].reset_index(drop=True)
            print(f"   Validation mode: using fold {fold}")

        # Filter for available files
        available_series = set(self.file_mapping.keys())
        self.df = self.df[self.df['SeriesInstanceUID'].isin(available_series)].reset_index(drop=True)
        
        print(f"   ✅ Dataset size: {len(self.df)} samples")

        # Cache data if requested
        if cache_data:
            print("\n💾 Caching data in memory...")
            for idx in tqdm(range(len(self.df)), desc="   Caching"):
                self._load_data(idx)
            print("   ✅ Caching complete!")

    def __len__(self) -> int:
        return len(self.df)

    def _load_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process a single sample."""
        series_id = self.df.iloc[idx]['SeriesInstanceUID']

        # Check cache
        if series_id in self.cache:
            return self.cache[series_id]

        npy_path = self.file_mapping.get(series_id)
        if npy_path is None:
            print(f"⚠️ File not found: {series_id}")
            # Return zeros with expected shape (channels, H, W)
            first_file = next(iter(self.file_mapping.values()))
            dummy_shape = np.load(first_file).shape[1:]  # (channels, H, W)
            return np.zeros(dummy_shape, dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)

        try:
            # Load 2.5D stacks: shape (num_stacks, channels, H, W)
            stacks = np.load(npy_path).astype(np.float32)
            
            # Validate shape
            if stacks.ndim != 4:
                print(f"⚠️ Unexpected shape {stacks.shape} for {series_id}, expected (N, C, H, W)")
                dummy_shape = (stacks.shape[1], stacks.shape[2], stacks.shape[3]) if stacks.ndim == 4 else (3, 256, 256)
                return np.zeros(dummy_shape, dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)
            
            num_stacks = stacks.shape[0]
            
            # Select stack based on strategy
            if self.stack_selection == 'middle' or num_stacks == 1:
                selected_stack = stacks[num_stacks // 2]
            elif self.stack_selection == 'random' and self.is_training:
                selected_stack = stacks[np.random.randint(0, num_stacks)]
            elif self.stack_selection == 'all':
                # Return all stacks (for inference/ensemble)
                selected_stack = stacks  # Shape: (num_stacks, channels, H, W)
            else:
                selected_stack = stacks[num_stacks // 2]
            
            # Get labels
            labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)

            # Cache if enabled
            if self.cache_data:
                self.cache[series_id] = (selected_stack, labels)

            return selected_stack, labels

        except Exception as e:
            print(f"❌ Error loading {series_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros with expected shape
            try:
                first_file = next(iter(self.file_mapping.values()))
                dummy_shape = np.load(first_file).shape[1:]  # (channels, H, W)
            except:
                dummy_shape = (3, 256, 256)  # Fallback
            return np.zeros(dummy_shape, dtype=np.float32), np.zeros(len(LABEL_COLS), dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        volume, labels = self._load_data(idx)

        X = torch.tensor(volume, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32)

        # Apply transforms (only during training)
        if self.transform and self.is_training:
            X = self.transform(X)

        return X, Y


def custom_collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    X_batch = torch.stack(images, dim=0)
    Y_batch = torch.stack(labels, dim=0)

    return X_batch, Y_batch
