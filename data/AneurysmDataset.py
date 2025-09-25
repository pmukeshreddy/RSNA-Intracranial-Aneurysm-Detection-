
class AneurysmDataset(Dataset):
    def __init__(self, npz_dir, train_df, fold, is_training, use_2d, slice_selection, cache_data, transform=None):
        self.npz_dir = Path(npz_dir)
        self.use_2d = use_2d
        self.slice_selection = slice_selection
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        
        # Create a mapping of series_id to full file path
        print("Building file index...")
        self.file_mapping = {}
        for root, dirs, files in os.walk(self.npz_dir):
            for f in files:
                if f.endswith('.npz'):
                    series_id = f.replace('.npz', '')
                    self.file_mapping[series_id] = Path(root) / f
        
        print(f"Found {len(self.file_mapping)} .npz files")

        if is_training:
            self.df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        else:
            self.df = train_df[train_df['fold'] == fold].reset_index(drop=True)

        # Filter dataframe to only include available files
        available_series = set(self.file_mapping.keys())
        self.df = self.df[self.df['SeriesInstanceUID'].isin(available_series)].reset_index(drop=True)
        
        print(f"Dataset size after filtering: {len(self.df)} samples")
        
        if cache_data:
            print("Caching all data in memory...")
            for idx in tqdm(range(len(self.df))):
                self._load_data(idx)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_data(self, idx):
        series_id = self.df.iloc[idx]['SeriesInstanceUID']

        if series_id in self.cache:
            return self.cache[series_id]

        # Get file path from mapping
        npz_path = self.file_mapping.get(series_id)
        
        if npz_path is None:
            print(f"File not found in mapping: {series_id}")
            return np.zeros((3, 128, 256, 256), dtype=np.float32), np.zeros(14, dtype=np.float32)
        
        try:
            data = np.load(npz_path, allow_pickle=True)

            # Get volume
            volume = data['final_input'].astype(np.float32)

            # Handle different shapes
            if volume.ndim == 3:  # (D, H, W)
                volume = volume[np.newaxis, ...]  # Add channel dimension: (1, D, H, W)

            # Standardize channels to 3
            if volume.shape[0] > 3:
                volume = volume[:3, ...]  # Take first 3 channels
            elif volume.shape[0] < 3:
                # Pad with zeros to get 3 channels
                padding = np.zeros((3 - volume.shape[0], *volume.shape[1:]), dtype=np.float32)
                volume = np.concatenate([volume, padding], axis=0)

            # Get labels
            labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)

            # Cache if enabled
            if self.cache_data:
                self.cache[series_id] = (volume, labels)

            return volume, labels

        except Exception as e:
            print(f"Error loading {series_id} from {npz_path}: {e}")
            # Return zeros with consistent shape
            return np.zeros((3, 128, 256, 256), dtype=np.float32), np.zeros(14, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load data
        volume, labels = self._load_data(idx)

        # Handle 2D vs 3D processing
        if self.use_2d:
            if volume.ndim == 4:
                volume = self._extract_slice(volume)
        else:
            # Handle variable depth sizes for 3D
            if volume.ndim == 4 and volume.shape[0] == 3:
                C, D, H, W = volume.shape
                target_depth = 128

                if D != target_depth:
                    # Resize depth dimension to consistent size
                    if D > target_depth:
                        # Crop: take center slices
                        start_idx = (D - target_depth) // 2
                        volume = volume[:, start_idx:start_idx + target_depth, :, :]
                    else:
                        # Pad: add zeros to reach target depth
                        pad_needed = target_depth - D
                        pad_before = pad_needed // 2
                        pad_after = pad_needed - pad_before

                        # Pad along depth dimension (dim=1)
                        volume = np.pad(volume, ((0, 0), (pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)

            else:
                # Fallback: reshape to expected size
                volume = volume.reshape(3, 128, 256, 256)

        # Clean tensor creation
        volume_array = np.array(volume, dtype=np.float32, copy=True)
        labels_array = np.array(labels, dtype=np.float32, copy=True)

        X = torch.tensor(volume_array, dtype=torch.float32)
        Y = torch.tensor(labels_array, dtype=torch.float32)

        # Verify final shape for 3D
        if not self.use_2d:
            assert X.shape == (3, 128, 256, 256), f"Unexpected shape: {X.shape} at index {idx}"

        # Apply transforms if needed
        if self.transform:
            X = self.transform(X)

        return X, Y

    def _extract_slice(self, volume: np.ndarray) -> np.ndarray:
        """Extract 2D slice(s) from 3D volume"""
        # volume shape: (C, D, H, W)
        n_channels, n_slices, height, width = volume.shape

        if self.slice_selection == 'middle':
            # Extract middle slice
            slice_idx = n_slices // 2
            return volume[:, slice_idx, :, :]  # (C, H, W)

        elif self.slice_selection == 'random':
            # Random slice (only in training)
            if hasattr(self, 'df'):  # Training mode
                slice_idx = random.randint(n_slices//4, 3*n_slices//4)  # Avoid edge slices
            else:
                slice_idx = n_slices // 2
            return volume[:, slice_idx, :, :]

        elif self.slice_selection == 'all':
            # Return all slices stacked (for multi-instance learning)
            # Reshape to (D, C, H, W) for batch processing
            return volume.transpose(1, 0, 2, 3)
            
        elif self.slice_selection == 'mip':
            # Create MIP by taking max along Depth axis
            return np.max(volume, axis=1)

        else:
            raise ValueError(f"Unknown slice_selection: {self.slice_selection}")
