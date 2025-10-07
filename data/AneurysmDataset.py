import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AneurysmDataset(Dataset):
    """
    Dataset for preprocessed aneurysm detection data
    Handles .npz files across multiple folders (part1, part2, part3, etc.)
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        data_folders: List[str],
        label_cols: List[str],
        transform=None,
        mode: str = 'train',
        cache_data: bool = False,
        max_slices: int = 64
    ):
        """
        Args:
            train_df: DataFrame with SeriesInstanceUID and labels
            data_folders: List of paths to folders containing .npz files
            label_cols: List of label column names
            transform: Albumentations transforms (optional)
            mode: 'train' or 'val'
            cache_data: Whether to cache loaded data in memory
            max_slices: Maximum number of slices to use (for memory efficiency)
        """
        self.train_df = train_df.reset_index(drop=True)
        self.data_folders = [Path(f) for f in data_folders]
        self.label_cols = label_cols
        self.transform = transform
        self.mode = mode
        self.cache_data = cache_data
        self.max_slices = max_slices
        
        # Build file path mapping
        self.file_map = self._build_file_map()
        
        # Filter df to only include series we have files for
        available_series = set(self.file_map.keys())
        self.train_df = self.train_df[
            self.train_df['SeriesInstanceUID'].isin(available_series)
        ].reset_index(drop=True)
        
        print(f"Initialized {mode} dataset:")
        print(f"  Total samples: {len(self.train_df)}")
        print(f"  Aneurysm positive: {self.train_df['Aneurysm Present'].sum()}")
        print(f"  Aneurysm rate: {self.train_df['Aneurysm Present'].mean():.2%}")
        
        # Optional: cache data in memory
        self.cache = {} if cache_data else None
        
    def _build_file_map(self) -> Dict[str, Path]:
        """Build mapping from SeriesInstanceUID to .npz file path"""
        file_map = {}
        
        for folder in self.data_folders:
            if not folder.exists():
                print(f"Warning: Folder not found: {folder}")
                continue
                
            npz_files = list(folder.glob("*.npz"))
            print(f"Found {len(npz_files)} files in {folder.name}")
            
            for npz_file in npz_files:
                series_id = npz_file.stem  # filename without .npz extension
                file_map[series_id] = npz_file
        
        print(f"Total mapped files: {len(file_map)}")
        return file_map
    
    def __len__(self) -> int:
        return len(self.train_df)
    
    def _load_npz(self, series_id: str) -> Tuple[np.ndarray, Dict]:
        """Load .npz file and return image data + metadata"""
        npz_path = self.file_map[series_id]
        
        with np.load(npz_path) as data:
            image_data = data['image_data']  # (3, D, H, W)
            modality = str(data['modality'])
            spacing = data['spacing']
        
        return image_data, {'modality': modality, 'spacing': spacing}
    
    def _process_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Process volume to fixed size
        Input: (3, D, H, W)
        Output: (3, max_slices, H, W)
        """
        C, D, H, W = volume.shape
        
        # Handle depth dimension
        if D > self.max_slices:
            # Sample evenly spaced slices
            indices = np.linspace(0, D-1, self.max_slices, dtype=int)
            volume = volume[:, indices, :, :]
        elif D < self.max_slices:
            # Pad with zeros
            pad_size = self.max_slices - D
            volume = np.pad(
                volume, 
                ((0, 0), (0, pad_size), (0, 0), (0, 0)), 
                mode='constant'
            )
        
        return volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.train_df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        
        # Check cache first
        if self.cache is not None and series_id in self.cache:
            volume, metadata = self.cache[series_id]
        else:
            volume, metadata = self._load_npz(series_id)
            
            if self.cache is not None:
                self.cache[series_id] = (volume, metadata)
        
        # Process volume to fixed size
        volume = self._process_volume(volume)  # (3, max_slices, H, W)
        
        # Apply transforms if in train mode
        if self.transform and self.mode == 'train':
            # Apply 2D transforms to each slice
            C, D, H, W = volume.shape
            transformed_slices = []
            
            for c in range(C):
                channel_slices = []
                for d in range(D):
                    slice_2d = volume[c, d, :, :]  # (H, W)
                    
                    # Albumentations expects (H, W, C) for images
                    # We'll treat each slice as single-channel
                    transformed = self.transform(image=slice_2d[..., np.newaxis])
                    channel_slices.append(transformed['image'].squeeze())
                
                transformed_slices.append(np.stack(channel_slices))
            
            volume = np.stack(transformed_slices)
        
        # Convert to torch tensor
        volume = torch.from_numpy(volume).float()
        
        # Get labels
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32)
        )
        
        return {
            'image': volume,  # (3, max_slices, H, W)
            'labels': labels,  # (num_labels,)
            'series_id': series_id,
            'modality': metadata['modality']
        }


# Data augmentation transforms
def get_train_transforms():
    """Training augmentations for 2D slices"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

def get_val_transforms():
    """Validation transforms (none, just for consistency)"""
    return None


# Example usage
def create_dataloaders(
    train_df: pd.DataFrame,
    data_folders: List[str],
    label_cols: List[str],
    fold: int = 0,
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a specific fold
    """
    
    # Split by fold
    train_df_fold = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_df_fold = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    # Create datasets
    train_dataset = AneurysmDataset(
        train_df=train_df_fold,
        data_folders=data_folders,
        label_cols=label_cols,
        transform=get_train_transforms(),
        mode='train',
        cache_data=False,  # Set True if you have enough RAM
        max_slices=64
    )
    
    val_dataset = AneurysmDataset(
        train_df=val_df_fold,
        data_folders=data_folders,
        label_cols=label_cols,
        transform=get_val_transforms(),
        mode='val',
        cache_data=False,
        max_slices=64
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    # Your data folders (update paths)
    data_folders = [
        '/path/to/part1',
        '/path/to/part2', 
        '/path/to/part3'
    ]
    
    # Label columns from your notebook
    label_cols = [
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
    
    # Create dataloaders for fold 0
    train_loader, val_loader = create_dataloaders(
        train_df=train_df_with_folds,  # Your df with folds
        data_folders=data_folders,
        label_cols=label_cols,
        fold=0,
        batch_size=4,
        num_workers=4
    )
    
    # Test loading
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")  # (B, 3, 64, H, W)
        print(f"Labels shape: {batch['labels'].shape}")  # (B, 14)
        print(f"Modality: {batch['modality']}")
        break
