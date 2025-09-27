class AneurysmDataset(Dataset):
    def __init__(self, npz_dir, train_df, fold, is_training, use_2d, slice_selection, cache_data, transform=None):
        self.npz_dir = Path(npz_dir)
        self.use_2d = use_2d
        self.slice_selection = slice_selection
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        self.is_training = is_training

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
            return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(14, dtype=np.float32)

        try:
            data = np.load(npz_path, allow_pickle=True)

            # Get volume - try different possible keys
            possible_keys = ['volume', 'final_input', 'data', 'image', 'array']
            volume = None
            available_keys = list(data.keys())

            for key in possible_keys:
                if key in available_keys:
                    volume = data[key]
                    break

            if volume is None:
                for key in available_keys:
                    if not key.startswith('__'):  # Skip metadata keys
                        volume = data[key]
                        print(f"Using fallback key '{key}' for {series_id}")
                        break

            if volume is None:
                print(f"No valid data key found in {series_id}. Available keys: {available_keys}")
                return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(14, dtype=np.float32)

            # Ensure float32 for processing
            volume = volume.astype(np.float32)

            # Handle 3D volume (D, H, W) - no channel dimension yet
            if volume.ndim == 3:
                pass  # This is expected
            elif volume.ndim == 4:
                # If already has channels, take first channel
                if volume.shape[0] <= 3:
                    volume = volume[0]
                else:
                    volume = volume[0]
            else:
                print(f"Unexpected volume shape: {volume.shape}")
                return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(14, dtype=np.float32)

            # DATA IS ALREADY NORMALIZED [0,1] - CREATE VARIETY BETWEEN CHANNELS
            # Check if data is normalized
            if volume.min() >= 0 and volume.max() <= 1.0:
                # Data is normalized, create different views for diversity
                
                # Channel 1: Original normalized data
                channel1 = volume
                
                # Channel 2: Enhanced contrast (stretch middle values)
                channel2 = np.clip(volume * 1.5 - 0.25, 0, 1)
                
                # Channel 3: Emphasis on bright regions (squared for contrast)
                channel3 = volume ** 2
                
                # Add slight random augmentation during training
                if self.is_training:
                    noise_factor = 0.02
                    channel1 = channel1 + np.random.normal(0, noise_factor, channel1.shape)
                    channel1 = np.clip(channel1, 0, 1)
                
            else:
                # Data is in HU units, apply standard windowing
                print(f"Data appears to be in HU units: [{volume.min():.1f}, {volume.max():.1f}]")
                
                # Brain window
                brain_window = np.clip(volume, -100, 300)
                channel1 = (brain_window + 100) / 400
                
                # Vessel window
                vessel_window = np.clip(volume, 40, 400)
                channel2 = (vessel_window - 40) / 360
                
                # Bone window
                bone_window = np.clip(volume, 200, 700)
                channel3 = (bone_window - 200) / 500

            # Stack as channels (3, D, H, W)
            volume = np.stack([channel1, channel2, channel3], axis=0)

            # Get labels
            labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)

            # Cache if enabled
            if self.cache_data:
                self.cache[series_id] = (volume, labels)

            return volume, labels

        except Exception as e:
            print(f"Error loading {series_id} from {npz_path}: {e}")
            return np.zeros((3, 64, 128, 128), dtype=np.float32), np.zeros(14, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load data
        volume, labels = self._load_data(idx)

        # Handle 2D vs 3D processing
        if self.use_2d:
            if volume.ndim == 4:
                volume = self._extract_slice(volume)
        else:
            # Handle variable sizes for 3D - resize ALL dimensions
            if volume.ndim == 4 and volume.shape[0] == 3:
                C, D, H, W = volume.shape
                target_depth = 64
                target_height = 128
                target_width = 128

                # Handle Depth dimension
                if D != target_depth:
                    if D > target_depth:
                        # Crop: take center slices
                        start_idx = (D - target_depth) // 2
                        volume = volume[:, start_idx:start_idx + target_depth, :, :]
                    else:
                        # Pad: add zeros to reach target depth
                        pad_needed = target_depth - D
                        pad_before = pad_needed // 2
                        pad_after = pad_needed - pad_before
                        volume = np.pad(
                            volume,
                            ((0, 0), (pad_before, pad_after), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )

                # Update dimensions after depth processing
                C, D, H, W = volume.shape

                # Handle Height dimension
                if H != target_height:
                    if H > target_height:
                        # Crop: take center
                        start_idx = (H - target_height) // 2
                        volume = volume[:, :, start_idx:start_idx + target_height, :]
                    else:
                        # Pad: add zeros
                        pad_needed = target_height - H
                        pad_before = pad_needed // 2
                        pad_after = pad_needed - pad_before
                        volume = np.pad(
                            volume,
                            ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )

                # Update dimensions after height processing
                C, D, H, W = volume.shape

                # Handle Width dimension
                if W != target_width:
                    if W > target_width:
                        # Crop: take center
                        start_idx = (W - target_width) // 2
                        volume = volume[:, :, :, start_idx:start_idx + target_width]
                    else:
                        # Pad: add zeros
                        pad_needed = target_width - W
                        pad_before = pad_needed // 2
                        pad_after = pad_needed - pad_before
                        volume = np.pad(
                            volume,
                            ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)),
                            mode='constant',
                            constant_values=0
                        )

            else:
                # Fallback: force resize to expected size
                print(f"Warning: Unexpected shape at index {idx}, forcing resize")
                temp_tensor = torch.tensor(volume, dtype=torch.float32)
                if temp_tensor.ndim == 3:
                    temp_tensor = temp_tensor.unsqueeze(0)
                if temp_tensor.ndim == 4:
                    if temp_tensor.shape[0] < 3:
                        padding = torch.zeros(3 - temp_tensor.shape[0], *temp_tensor.shape[1:], dtype=torch.float32)
                        temp_tensor = torch.cat([temp_tensor, padding], dim=0)
                    elif temp_tensor.shape[0] > 3:
                        temp_tensor = temp_tensor[:3]

                    temp_tensor = temp_tensor.unsqueeze(0)
                    temp_tensor = torch.nn.functional.interpolate(
                        temp_tensor, size=(64,128,128), mode='trilinear', align_corners=False
                    )
                    volume = temp_tensor.squeeze(0).numpy()

        # Clean tensor creation
        volume_array = np.array(volume, dtype=np.float32, copy=True)
        labels_array = np.array(labels, dtype=np.float32, copy=True)

        X = torch.tensor(volume_array, dtype=torch.float32)
        Y = torch.tensor(labels_array, dtype=torch.float32)

        # Verify final shape for 3D
        if not self.use_2d:
            if X.shape != (3, 64,128,128):
                print(f"Shape mismatch at index {idx}: {X.shape}")
                X = X.unsqueeze(0)
                X = torch.nn.functional.interpolate(
                    X, size=(64,128,128), mode='trilinear', align_corners=False
                )
                X = X.squeeze(0)

        # Apply transforms if needed
        if self.transform:
            X = self.transform(X)

        return X, Y

    def _extract_slice(self, volume: np.ndarray) -> np.ndarray:
        """Extract 2D slice(s) from 3D volume"""
        n_channels, n_slices, height, width = volume.shape

        if self.slice_selection == 'middle':
            slice_idx = n_slices // 2
            return volume[:, slice_idx, :, :]

        elif self.slice_selection == 'random':
            if hasattr(self, 'df'):
                slice_idx = random.randint(n_slices//4, 3*n_slices//4)
            else:
                slice_idx = n_slices // 2
            return volume[:, slice_idx, :, :]

        elif self.slice_selection == 'all':
            return volume.transpose(1, 0, 2, 3)

        elif self.slice_selection == 'mip':
            return np.max(volume, axis=1)

        else:
            raise ValueError(f"Unknown slice_selection: {self.slice_selection}")
