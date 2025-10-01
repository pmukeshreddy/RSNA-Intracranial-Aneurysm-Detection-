import pydicom
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, Optional, Dict, List

class DICOMPreprocessor:
    """
    Unified preprocessing pipeline for CT and MR/MRA scans with 2.5D support.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        ct_window_center: float = 40.0,
        ct_window_width: float = 450.0,
        mr_percentile_range: Tuple[float, float] = (1.0, 99.0),
        use_dicom_window_for_mr: bool = True,
        num_slices_2_5d: int = 3
    ):
        """
        Args:
            target_size: Output image dimensions (H, W)
            ct_window_center: HU window center for CT scans
            ct_window_width: HU window width for CT scans
            mr_percentile_range: Percentile clipping for MR (min, max)
            use_dicom_window_for_mr: Use DICOM window settings if available for MR
            num_slices_2_5d: Number of slices for 2.5D (usually 3 or 5)
        """
        self.target_size = target_size
        self.ct_window_center = ct_window_center
        self.ct_window_width = ct_window_width
        self.mr_percentile_range = mr_percentile_range
        self.use_dicom_window_for_mr = use_dicom_window_for_mr
        self.num_slices_2_5d = num_slices_2_5d
    
    def load_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM and extract metadata."""
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        metadata = {
            'modality': getattr(dcm, 'Modality', None),
            'rescale_slope': getattr(dcm, 'RescaleSlope', None),
            'rescale_intercept': getattr(dcm, 'RescaleIntercept', None),
            'window_center': getattr(dcm, 'WindowCenter', None),
            'window_width': getattr(dcm, 'WindowWidth', None),
            'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
            'series_description': getattr(dcm, 'SeriesDescription', None),
            'original_shape': pixel_array.shape,
            'instance_number': getattr(dcm, 'InstanceNumber', None),
            'image_position': getattr(dcm, 'ImagePositionPatient', None),
            'slice_location': getattr(dcm, 'SliceLocation', None)
        }
        
        # Handle multi-value window settings
        if isinstance(metadata['window_center'], (list, pydicom.multival.MultiValue)):
            metadata['window_center'] = float(metadata['window_center'][0])
        elif metadata['window_center'] is not None:
            metadata['window_center'] = float(metadata['window_center'])
            
        if isinstance(metadata['window_width'], (list, pydicom.multival.MultiValue)):
            metadata['window_width'] = float(metadata['window_width'][0])
        elif metadata['window_width'] is not None:
            metadata['window_width'] = float(metadata['window_width'])
        
        return pixel_array, metadata
    
    def load_series(self, series_path: str, verbose: bool = True) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """
        Load all DICOM files from a series directory.
        
        Returns:
            pixel_arrays: List of pixel arrays
            metadata_list: List of metadata dicts
            file_paths: List of file paths
        """
        series_dir = Path(series_path)
        dicom_files = sorted(series_dir.glob('*.dcm'))
        
        if verbose:
            print(f"\n📂 Loading series: {series_dir.name}")
            print(f"   Found {len(dicom_files)} DICOM files")
        
        pixel_arrays = []
        metadata_list = []
        file_paths = []
        
        for i, dcm_file in enumerate(dicom_files, 1):
            if verbose and i % 10 == 0:
                print(f"   Loading... {i}/{len(dicom_files)}")
            
            pixel_array, metadata = self.load_dicom(str(dcm_file))
            pixel_arrays.append(pixel_array)
            metadata_list.append(metadata)
            file_paths.append(str(dcm_file))
        
        if verbose:
            print(f"   ✅ Loaded {len(pixel_arrays)} slices")
        
        return pixel_arrays, metadata_list, file_paths
    
    def sort_slices(
        self, 
        pixel_arrays: List[np.ndarray], 
        metadata_list: List[Dict],
        file_paths: List[str],
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """
        Sort slices by position (using ImagePositionPatient Z-coordinate or InstanceNumber).
        """
        if verbose:
            print(f"\n🔄 Sorting slices by spatial position...")
        
        # Try to sort by ImagePositionPatient (Z coordinate)
        positions = []
        for metadata in metadata_list:
            if metadata['image_position'] is not None:
                positions.append(float(metadata['image_position'][2]))  # Z coordinate
            elif metadata['slice_location'] is not None:
                positions.append(float(metadata['slice_location']))
            elif metadata['instance_number'] is not None:
                positions.append(float(metadata['instance_number']))
            else:
                positions.append(0)  # Fallback
        
        # Sort by position
        sorted_indices = np.argsort(positions)
        
        pixel_arrays = [pixel_arrays[i] for i in sorted_indices]
        metadata_list = [metadata_list[i] for i in sorted_indices]
        file_paths = [file_paths[i] for i in sorted_indices]
        
        if verbose:
            print(f"   ✅ Sorted {len(pixel_arrays)} slices")
        
        return pixel_arrays, metadata_list, file_paths
    
    def apply_ct_windowing(self, pixel_array: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply HU windowing for CT scans."""
        slope = metadata['rescale_slope']
        intercept = metadata['rescale_intercept']
        
        if slope is not None and intercept is not None:
            hu_array = pixel_array * slope + intercept
        else:
            hu_array = pixel_array
        
        window_min = self.ct_window_center - (self.ct_window_width / 2)
        window_max = self.ct_window_center + (self.ct_window_width / 2)
        
        windowed = np.clip(hu_array, window_min, window_max)
        normalized = (windowed - window_min) / (window_max - window_min)
        
        return normalized
    
    def apply_mr_normalization(self, pixel_array: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply normalization for MR/MRA scans."""
        if self.use_dicom_window_for_mr and metadata['window_center'] is not None:
            center = metadata['window_center']
            width = metadata['window_width']
            
            window_min = center - (width / 2)
            window_max = center + (width / 2)
            
            windowed = np.clip(pixel_array, window_min, window_max)
            normalized = (windowed - window_min) / (window_max - window_min)
        else:
            p_min, p_max = self.mr_percentile_range
            vmin = np.percentile(pixel_array, p_min)
            vmax = np.percentile(pixel_array, p_max)
            
            clipped = np.clip(pixel_array, vmin, vmax)
            normalized = (clipped - vmin) / (vmax - vmin + 1e-8)
        
        return normalized
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[:2] != self.target_size:
            resized = cv2.resize(
                image, 
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            return resized
        return image
    
    def preprocess_single_slice(self, pixel_array: np.ndarray, metadata: Dict) -> np.ndarray:
        """Preprocess a single slice."""
        modality = metadata['modality']
        
        if modality == 'CT':
            processed = self.apply_ct_windowing(pixel_array, metadata)
        elif modality in ['MR', 'MRA']:
            processed = self.apply_mr_normalization(pixel_array, metadata)
        else:
            vmin = np.percentile(pixel_array, 1)
            vmax = np.percentile(pixel_array, 99)
            processed = np.clip(pixel_array, vmin, vmax)
            processed = (processed - vmin) / (vmax - vmin + 1e-8)
        
        processed = self.resize_image(processed)
        return processed
    
    def create_2_5d_stacks(
        self,
        series_path: str,
        return_metadata: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Create 2.5D stacks from a series directory.
        
        Args:
            series_path: Path to series directory containing DICOM files
            return_metadata: If True, return (stacks, metadata_list)
            verbose: Print progress
            
        Returns:
            stacks: numpy array of shape (num_slices, num_channels, H, W)
                   where num_channels = num_slices_2_5d (e.g., 3)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔧 CREATING 2.5D STACKS")
            print(f"{'='*60}")
        
        # Load and sort all slices
        pixel_arrays, metadata_list, file_paths = self.load_series(series_path, verbose=verbose)
        pixel_arrays, metadata_list, file_paths = self.sort_slices(
            pixel_arrays, metadata_list, file_paths, verbose=verbose
        )
        
        num_slices = len(pixel_arrays)
        half_window = self.num_slices_2_5d // 2
        
        if verbose:
            print(f"\n⚙️  Preprocessing {num_slices} slices...")
            modality = metadata_list[0]['modality']
            print(f"   Modality: {modality}")
            print(f"   2.5D window: {self.num_slices_2_5d} slices")
        
        # Preprocess all slices
        processed_slices = []
        for i in range(num_slices):
            if verbose and (i + 1) % 20 == 0:
                print(f"   Preprocessing... {i+1}/{num_slices}")
            
            processed = self.preprocess_single_slice(pixel_arrays[i], metadata_list[i])
            processed_slices.append(processed)
        
        if verbose:
            print(f"   ✅ Preprocessed {num_slices} slices")
        
        # Create 2.5D stacks
        if verbose:
            print(f"\n📚 Creating 2.5D stacks...")
        
        stacks = []
        stack_metadata = []
        
        for i in range(num_slices):
            if verbose and (i + 1) % 20 == 0:
                print(f"   Stacking... {i+1}/{num_slices}")
            
            # Get neighboring slices
            stack_slices = []
            
            for offset in range(-half_window, half_window + 1):
                idx = i + offset
                
                # Handle edge cases with padding (repeat first/last slice)
                if idx < 0:
                    idx = 0
                elif idx >= num_slices:
                    idx = num_slices - 1
                
                stack_slices.append(processed_slices[idx])
            
            # Stack along channel dimension
            stack = np.stack(stack_slices, axis=0)  # Shape: (num_channels, H, W)
            stacks.append(stack)
            
            if return_metadata:
                stack_metadata.append({
                    'center_slice_idx': i,
                    'center_slice_path': file_paths[i],
                    'metadata': metadata_list[i],
                    'slice_indices': [max(0, min(num_slices-1, i+offset)) 
                                     for offset in range(-half_window, half_window + 1)]
                })
        
        stacks = np.array(stacks)
        
        if verbose:
            print(f"   ✅ Created {len(stacks)} stacks")
            print(f"\n📊 Output shape: {stacks.shape}")
            print(f"   (num_stacks, channels, height, width)")
            print(f"{'='*60}\n")
        
        if return_metadata:
            return stacks, stack_metadata
        return stacks
    
    def preprocess(self, dicom_path: str, return_metadata: bool = False) -> np.ndarray:
        """
        Preprocess single DICOM file (for backward compatibility).
        
        Returns:
            Preprocessed image normalized to [0, 1] with shape target_size
        """
        pixel_array, metadata = self.load_dicom(dicom_path)
        processed = self.preprocess_single_slice(pixel_array, metadata)
        
        if return_metadata:
            return processed, metadata
        return processed


def visualize_2_5d_stack(series_path: str, slice_idx: int, preprocessor: DICOMPreprocessor):
    """
    Visualize a 2.5D stack.
    
    Args:
        series_path: Path to series directory
        slice_idx: Index of the slice to visualize
        preprocessor: DICOMPreprocessor instance
    """
    import matplotlib.pyplot as plt
    
    stacks, metadata_list = preprocessor.create_2_5d_stacks(
        series_path, 
        return_metadata=True
    )
    
    if slice_idx >= len(stacks):
        slice_idx = len(stacks) // 2
    
    stack = stacks[slice_idx]  # Shape: (3, H, W)
    meta = metadata_list[slice_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channel_names = ['Previous Slice', 'Current Slice', 'Next Slice']
    
    for i in range(3):
        axes[i].imshow(stack[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'{channel_names[i]}\n(Index: {meta["slice_indices"][i]})')
        axes[i].axis('off')
    
    plt.suptitle(f'2.5D Stack - Center Slice: {slice_idx}\n'
                 f'Modality: {meta["metadata"]["modality"]}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"\n2.5D Stack Info:")
    print(f"Shape: {stack.shape}")
    print(f"Center slice: {slice_idx}")
    print(f"Total stacks in series: {len(stacks)}")
    print(f"Modality: {meta['metadata']['modality']}")


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DICOMPreprocessor(
        target_size=(512, 512),
        ct_window_center=40.0,
        ct_window_width=450.0,
        num_slices_2_5d=3  # Use 3 slices for 2.5D
    )
    
    # Create 2.5D stacks from a series
    # series_path = '/kaggle/input/.../train/patient_id/series_id/'
    # stacks = preprocessor.create_2_5d_stacks(series_path, verbose=True)
    # print(f"Created {len(stacks)} 2.5D stacks with shape: {stacks[0].shape}")
    
    # With metadata
    # stacks, metadata_list = preprocessor.create_2_5d_stacks(
    #     series_path,
    #     return_metadata=True,
    #     verbose=True
    # )
    
    # Visualize a stack
    # visualize_2_5d_stack(series_path, slice_idx=20, preprocessor=preprocessor)
    
    # Process all series in training data
    # train_path = Path('/kaggle/input/.../train')
    # for patient_dir in train_path.iterdir():
    #     print(f"\n🏥 Processing patient: {patient_dir.name}")
    #     for series_dir in patient_dir.iterdir():
    #         print(f"   📁 Series: {series_dir.name}")
    #         stacks = preprocessor.create_2_5d_stacks(str(series_dir), verbose=True)
    #         # Save or process stacks
    #         np.save(f'processed/{series_dir.name}.npy', stacks)
    #         print(f"   💾 Saved: processed/{series_dir.name}.npy")
    
    pass
