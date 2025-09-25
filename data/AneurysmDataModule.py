class AneurysmDataModule:
    """Data module to handle train/val/test splits and dataloaders"""

    def __init__(self,
                 npz_dir: str,
                 train_csv: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 fold: int = 0,
                 use_2d: bool = True,
                 cache_data: bool = False):

        self.npz_dir = Path(npz_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.use_2d = use_2d
        self.cache_data = cache_data

        # Load dataframe with labels
        self.train_df = pd.read_csv(train_csv)

        # Add fold column if not present
        if 'fold' not in self.train_df.columns:
            print("Creating folds...")
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(
                self.train_df,
                self.train_df['Aneurysm Present']
            )):
                self.train_df.loc[val_idx, 'fold'] = fold

        # Define augmentations
        self.train_transform = get_train_transforms()
        
        # Create datasets
        self.train_dataset = AneurysmDataset(
            npz_dir=npz_dir,
            train_df=self.train_df,
            fold=fold,
            is_training=True,
            use_2d=use_2d,
            slice_selection='middle' if use_2d else None,
            cache_data=cache_data,
            transform=self.train_transform
        )

        self.val_dataset = AneurysmDataset(
            npz_dir=npz_dir,
            train_df=self.train_df,
            fold=fold,
            is_training=False,
            use_2d=use_2d,
            slice_selection='middle' if use_2d else None,
            cache_data=cache_data
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

    def get_label_weights(self) -> torch.Tensor:
        # Calculate positive weights for each label
        pos_counts = self.train_df[LABEL_COLS].sum()
        neg_counts = len(self.train_df) - pos_counts

        # Weight = neg_count / pos_count (higher weight for rare classes)
        weights = neg_counts / (pos_counts + 1)  # +1 to avoid division by zero

        # Cap weights to avoid instability
        weights = np.clip(weights, 1, 20)

        # Special handling for 'Aneurysm Present' (most important)
        weights.iloc[-1] = weights.iloc[-1] * 2

        return torch.FloatTensor(weights.values)
