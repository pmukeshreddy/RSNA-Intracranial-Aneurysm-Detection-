import pydicom
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, Optional, Dict, List
import shutil
import zipfile
import os

class DICOMPreprocessor:
    """
    Unified preprocessing pipeline for CT and MR/MRA scans with 2.5D support.
    Handles both multi-frame DICOMs and multiple single-frame DICOM files.
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
    
    def load_dicom(self, dicom_path: str, verbose: bool = False) -> Tuple[List[np.ndarray], Dict]:
        """
        Load DICOM and extract all frames (handles multi-frame).
        
        Returns:
            pixel_arrays: List of pixel arrays (one per frame)
            metadata: Metadata dict
        """
        dcm = pydicom.dcmread(dicom_path)
        
        # Check if multi-frame
        pixel_array = dcm.pixel_array
        
        # Convert to list of frames
        if len(pixel_array.shape) == 2:
            # Single frame
            pixel_arrays = [pixel_array.astype(np.float32)]
            num_frames = 1
        elif len(pixel_array.shape) == 3:
            # Multi-frame: shape is (num_frames, height, width)
            pixel_arrays = [pixel_array[i].astype(np.float32) for i in range(pixel_array.shape[0])]
            num_frames = pixel_array.shape[0]
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")
        
        metadata = {
            'modality': getattr(dcm, 'Modality', None),
            'rescale_slope': getattr(dcm, 'RescaleSlope', None),
            'rescale_intercept': getattr(dcm, 'RescaleIntercept', None),
            'window_center': getattr(dcm, 'WindowCenter', None),
            'window_width': getattr(dcm, 'WindowWidth', None),
            'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
            'series_description': getattr(dcm, 'SeriesDescription', None),
            'original_shape': pixel_arrays[0].shape,
            'num_frames': num_frames,
            'is_multiframe': num_frames > 1,
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
        
        if verbose:
            frame_type = "multi-frame" if num_frames > 1 else "single-frame"
            print(f"   Loaded {frame_type} DICOM with {num_frames} frame(s)")
        
        return pixel_arrays, metadata
    
    def load_patient_dicoms(
        self, 
        patient_path: str, 
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """
        Load all DICOM files from a patient directory.
        Handles both multi-frame and single-frame DICOMs.
        
        Returns:
            all_frames: List of all pixel arrays (flattened across all DICOMs)
            metadata_list: List of metadata dicts (one per frame)
            frame_sources: List of source info (which DICOM each frame came from)
        """
        patient_dir = Path(patient_path)
        
        if not patient_dir.is_dir():
            if verbose:
                print(f"\n⚠️  Path is not a directory: {patient_path}")
            return [], [], []
        
        dicom_files = sorted(patient_dir.glob('*.dcm'))
        
        if verbose:
            print(f"\n📂 Loading patient: {patient_dir.name}")
            print(f"   Found {len(dicom_files)} DICOM file(s)")
        
        if len(dicom_files) == 0:
            if verbose:
                print(f"   ⚠️  No DICOM files found")
            return [], [], []
        
        all_frames = []
        metadata_list = []
        frame_sources = []
        
        for i, dcm_file in enumerate(dicom_files, 1):
            if verbose and i % 5 == 0:
                print(f"   Loading... {i}/{len(dicom_files)}")
            
            try:
                pixel_arrays, metadata = self.load_dicom(str(dcm_file), verbose=False)
                
                # Add all frames from this DICOM
                for frame_idx, pixel_array in enumerate(pixel_arrays):
                    all_frames.append(pixel_array)
                    
                    # Create metadata for this frame
                    frame_metadata = metadata.copy()
                    frame_metadata['source_file'] = dcm_file.name
                    frame_metadata['frame_index'] = frame_idx
                    metadata_list.append(frame_metadata)
                    
                    frame_sources.append({
                        'file': str(dcm_file),
                        'frame_idx': frame_idx,
                        'total_frames': metadata['num_frames']
                    })
            
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error loading {dcm_file.name}: {e}")
                continue
        
        if verbose:
            print(f"   ✅ Loaded {len(all_frames)} total frame(s) from {len(dicom_files)} file(s)")
        
        return all_frames, metadata_list, frame_sources
    
    def sort_frames(
        self,
        pixel_arrays: List[np.ndarray],
        metadata_list: List[Dict],
        frame_sources: List[str],
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """Sort frames by spatial position."""
        if len(pixel_arrays) == 0:
            return [], [], []
        
        if verbose:
            print(f"\n🔄 Sorting frames by spatial position...")
        
        # Try to sort by position
        positions = []
        for metadata in metadata_list:
            if metadata['image_position'] is not None:
                positions.append(float(metadata['image_position'][2]))
            elif metadata['slice_location'] is not None:
                positions.append(float(metadata['slice_location']))
            elif metadata['instance_number'] is not None:
                positions.append(float(metadata['instance_number']))
            else:
                positions.append(0)
        
        sorted_indices = np.argsort(positions)
        
        pixel_arrays = [pixel_arrays[i] for i in sorted_indices]
        metadata_list = [metadata_list[i] for i in sorted_indices]
        frame_sources = [frame_sources[i] for i in sorted_indices]
        
        if verbose:
            print(f"   ✅ Sorted {len(pixel_arrays)} frame(s)")
        
        return pixel_arrays, metadata_list, frame_sources
    
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
    
    def preprocess_single_frame(self, pixel_array: np.ndarray, metadata: Dict) -> np.ndarray:
        """Preprocess a single frame."""
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
        patient_path: str,
        return_metadata: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Create 2.5D stacks from a patient directory.
        Handles multi-frame DICOMs and multiple single-frame DICOMs.
        
        Args:
            patient_path: Path to patient directory containing DICOM files
            return_metadata: If True, return (stacks, metadata_list)
            verbose: Print progress
            
        Returns:
            stacks: numpy array of shape (num_stacks, num_channels, H, W)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔧 CREATING 2.5D STACKS")
            print(f"{'='*60}")
        
        # Load all frames from all DICOMs
        pixel_arrays, metadata_list, frame_sources = self.load_patient_dicoms(
            patient_path, 
            verbose=verbose
        )
        
        if len(pixel_arrays) == 0:
            if verbose:
                print(f"\n⚠️  No frames loaded. Skipping.")
                print(f"{'='*60}\n")
            if return_metadata:
                return np.array([]), []
            return np.array([])
        
        # Sort frames spatially
        pixel_arrays, metadata_list, frame_sources = self.sort_frames(
            pixel_arrays, metadata_list, frame_sources, verbose=verbose
        )
        
        num_frames = len(pixel_arrays)
        half_window = self.num_slices_2_5d // 2
        
        if verbose:
            print(f"\n⚙️  Preprocessing {num_frames} frame(s)...")
            modality = metadata_list[0]['modality']
            print(f"   Modality: {modality}")
            print(f"   2.5D window: {self.num_slices_2_5d} slices")
        
        # Preprocess all frames
        processed_frames = []
        for i in range(num_frames):
            if verbose and (i + 1) % 20 == 0:
                print(f"   Preprocessing... {i+1}/{num_frames}")
            
            processed = self.preprocess_single_frame(pixel_arrays[i], metadata_list[i])
            processed_frames.append(processed)
        
        if verbose:
            print(f"   ✅ Preprocessed {num_frames} frame(s)")
        
        # Create 2.5D stacks
        if verbose:
            print(f"\n📚 Creating 2.5D stacks...")
        
        stacks = []
        stack_metadata = []
        
        for i in range(num_frames):
            if verbose and (i + 1) % 20 == 0:
                print(f"   Stacking... {i+1}/{num_frames}")
            
            # Get neighboring frames
            stack_frames = []
            
            for offset in range(-half_window, half_window + 1):
                idx = i + offset
                
                # Handle edge cases (repeat first/last frame)
                if idx < 0:
                    idx = 0
                elif idx >= num_frames:
                    idx = num_frames - 1
                
                stack_frames.append(processed_frames[idx])
            
            # Stack along channel dimension
            stack = np.stack(stack_frames, axis=0)  # Shape: (num_channels, H, W)
            stacks.append(stack)
            
            if return_metadata:
                stack_metadata.append({
                    'center_frame_idx': i,
                    'source_info': frame_sources[i],
                    'metadata': metadata_list[i],
                    'frame_indices': [max(0, min(num_frames-1, i+offset)) 
                                     for offset in range(-half_window, half_window + 1)]
                })
        
        stacks = np.array(stacks)
        
        if verbose:
            print(f"   ✅ Created {len(stacks)} stack(s)")
            print(f"\n📊 Output shape: {stacks.shape}")
            print(f"   (num_stacks, channels, height, width)")
            print(f"{'='*60}\n")
        
        if return_metadata:
            return stacks, stack_metadata
        return stacks
    
    def preprocess_single_dicom(
        self,
        dicom_path: str,
        return_metadata: bool = False,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Preprocess a single DICOM file (for 2D approach or when not using 2.5D).
        If multi-frame, returns all frames preprocessed.
        
        Returns:
            Processed frames with shape (num_frames, H, W)
        """
        pixel_arrays, metadata = self.load_dicom(dicom_path, verbose=verbose)
        
        processed_frames = []
        for pixel_array in pixel_arrays:
            processed = self.preprocess_single_frame(pixel_array, metadata)
            processed_frames.append(processed)
        
        processed_frames = np.array(processed_frames)
        
        if return_metadata:
            return processed_frames, metadata
        return processed_frames


def visualize_2_5d_stack(
    patient_path: str, 
    stack_idx: int, 
    preprocessor: DICOMPreprocessor
):
    """Visualize a 2.5D stack."""
    import matplotlib.pyplot as plt
    
    stacks, metadata_list = preprocessor.create_2_5d_stacks(
        patient_path, 
        return_metadata=True,
        verbose=False
    )
    
    if len(stacks) == 0:
        print("No stacks to visualize!")
        return
    
    if stack_idx >= len(stacks):
        stack_idx = len(stacks) // 2
    
    stack = stacks[stack_idx]
    meta = metadata_list[stack_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channel_names = ['Previous Frame', 'Current Frame', 'Next Frame']
    
    for i in range(3):
        axes[i].imshow(stack[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'{channel_names[i]}\n(Index: {meta["frame_indices"][i]})')
        axes[i].axis('off')
    
    plt.suptitle(f'2.5D Stack - Center Frame: {stack_idx}\n'
                 f'Modality: {meta["metadata"]["modality"]}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"\n2.5D Stack Info:")
    print(f"Shape: {stack.shape}")
    print(f"Center frame: {stack_idx}")
    print(f"Total stacks: {len(stacks)}")
    print(f"Source: {meta['source_info']}")


def get_directory_size_gb(directory: str) -> float:
    """Get total size of directory in GB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 ** 3)  # Convert to GB


def zip_and_cleanup(
    working_dir: str,
    batch_num: int,
    download: bool = True,
    verbose: bool = True
) -> str:
    """
    Zip all .npy files in working directory and optionally clean up.
    
    Args:
        working_dir: Directory containing .npy files
        batch_num: Batch number for naming
        download: If True, print download instructions
        verbose: Print progress
    
    Returns:
        Path to created zip file
    """
    working_path = Path(working_dir)
    zip_filename = f'preprocessed_batch_{batch_num}.zip'
    zip_path = working_path / zip_filename
    
    npy_files = list(working_path.glob('*.npy'))
    
    if len(npy_files) == 0:
        if verbose:
            print("⚠️  No .npy files to zip!")
        return None
    
    if verbose:
        print(f"\n📦 Zipping {len(npy_files)} file(s)...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, npy_file in enumerate(npy_files, 1):
            if verbose and i % 10 == 0:
                print(f"   Zipping... {i}/{len(npy_files)}")
            zipf.write(npy_file, npy_file.name)
    
    zip_size_gb = os.path.getsize(zip_path) / (1024 ** 3)
    
    if verbose:
        print(f"   ✅ Created: {zip_filename} ({zip_size_gb:.2f} GB)")
    
    if download:
        if verbose:
            print(f"\n⬇️  DOWNLOAD THIS FILE NOW:")
            print(f"   Right-click → Download: {zip_filename}")
            print(f"   Or use: files.download('{zip_path}')")
    
    # Delete individual .npy files to free space
    if verbose:
        print(f"\n🗑️  Cleaning up {len(npy_files)} .npy file(s)...")
    
    for npy_file in npy_files:
        npy_file.unlink()
    
    if verbose:
        print(f"   ✅ Cleaned up, freed space!")
        print(f"   Keep {zip_filename} - download it before continuing!\n")
    
    return str(zip_path)


def process_dataset_with_auto_zip(
    train_path: str,
    preprocessor: DICOMPreprocessor,
    output_dir: str = '/kaggle/working',
    size_threshold_gb: float = 19.0,
    verbose: bool = True
):
    """
    Process entire dataset with automatic zipping when storage limit approached.
    
    Args:
        train_path: Path to training data directory
        preprocessor: DICOMPreprocessor instance
        output_dir: Output directory for processed files
        size_threshold_gb: Zip and cleanup when this size is reached (GB)
        verbose: Print progress
    """
    train_dir = Path(train_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    patient_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    if verbose:
        print(f"{'='*60}")
        print(f"🚀 BATCH PROCESSING WITH AUTO-ZIP")
        print(f"{'='*60}")
        print(f"Total patients: {len(patient_dirs)}")
        print(f"Storage threshold: {size_threshold_gb} GB")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    
    batch_num = 1
    processed_count = 0
    
    for i, patient_dir in enumerate(patient_dirs, 1):
        if verbose:
            print(f"\n[{i}/{len(patient_dirs)}] 🏥 Patient: {patient_dir.name}")
        
        try:
            # Process patient
            stacks = preprocessor.create_2_5d_stacks(
                str(patient_dir), 
                verbose=False
            )
            
            if len(stacks) > 0:
                output_file = output_path / f'{patient_dir.name}.npy'
                np.save(output_file, stacks)
                processed_count += 1
                
                if verbose:
                    print(f"   ✅ Saved: {patient_dir.name}.npy ({stacks.shape})")
            else:
                if verbose:
                    print(f"   ⚠️  Skipped (no frames)")
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            continue
        
        # Check storage after each patient
        current_size = get_directory_size_gb(output_dir)
        
        if verbose and processed_count % 5 == 0:
            print(f"\n💾 Storage: {current_size:.2f} GB / {size_threshold_gb} GB")
        
        # Auto-zip when threshold reached
        if current_size >= size_threshold_gb:
            if verbose:
                print(f"\n{'='*60}")
                print(f"⚠️  STORAGE THRESHOLD REACHED: {current_size:.2f} GB")
                print(f"{'='*60}")
            
            zip_path = zip_and_cleanup(
                output_dir,
                batch_num,
                download=True,
                verbose=verbose
            )
            
            if zip_path:
                batch_num += 1
                processed_count = 0
                
                if verbose:
                    print(f"{'='*60}")
                    print(f"✅ Batch {batch_num-1} complete! Continuing...")
                    print(f"{'='*60}\n")
    
    # Final zip for remaining files
    current_size = get_directory_size_gb(output_dir)
    if current_size > 0.1:  # If there are remaining files
        if verbose:
            print(f"\n{'='*60}")
            print(f"📦 FINAL BATCH ({current_size:.2f} GB)")
            print(f"{'='*60}")
        
        zip_and_cleanup(
            output_dir,
            batch_num,
            download=True,
            verbose=verbose
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎉 ALL PROCESSING COMPLETE!")
        print(f"   Total batches: {batch_num}")
        print(f"{'='*60}\n")


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    preprocessor = DICOMPreprocessor(
        target_size=(512, 512),
        ct_window_center=40.0,
        ct_window_width=450.0,
        num_slices_2_5d=3
    )
    
    # AUTO-ZIP BATCH PROCESSING (Recommended for large datasets)
    process_dataset_with_auto_zip(
         train_path='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
         preprocessor=preprocessor,
         output_dir='/kaggle/working',
         size_threshold_gb=19.0,  # Zip when reaching 19GB
         verbose=True
     )
    
    # Manual processing (old way)
    # train_path = Path('/kaggle/input/.../train')
    # for patient_dir in train_path.iterdir():
    #     if not patient_dir.is_dir():
    #         continue
    #     stacks = preprocessor.create_2_5d_stacks(str(patient_dir))
    #     if len(stacks) > 0:
    #         np.save(f'/kaggle/working/{patient_dir.name}.npy', stacks)
    
    pass
