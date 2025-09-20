import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm.auto import tqdm
import cv2
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

# Simple config
CONFIG = {
    'TRAIN_CSV': '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv',
    'SERIES_DIR': '/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    'OUTPUT_DIR': '/kaggle/working/preprocessed_simple',
    'TARGET_SIZE': (64, 128, 128),  # (depth, height, width)
    'BATCH_SIZE': 50  # Process in batches
}

# Load data
train_df = pd.read_csv(CONFIG['TRAIN_CSV'])
print(f"Total series: {len(train_df)}")

def load_dicom_volume(series_path):
    """Simple DICOM loader"""
    dicom_files = sorted(Path(series_path).glob("*.dcm"))
    if not dicom_files:
        return None, None
    
    slices = []
    for dcm_file in dicom_files[:100]:  # Limit slices to speed up
        try:
            ds = pydicom.dcmread(str(dcm_file))
            if hasattr(ds, 'pixel_array'):
                slices.append(ds.pixel_array.astype(np.float32))
        except:
            continue
    
    if not slices:
        return None, None
    
    volume = np.stack(slices, axis=0)
    
    # Apply simple windowing for CTA/MRA
    modality = dicom_files[0].parent.name  # Quick modality guess
    if volume.max() > 1000:  # Likely HU values
        # Simple vessel window
        volume = np.clip(volume, -100, 400)
        volume = (volume + 100) / 500
    else:
        # Simple normalization
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-6)
    
    return volume, {'shape': volume.shape}

def process_volume_simple(volume, target_size=CONFIG['TARGET_SIZE']):
    """Quick resize to target size"""
    if volume is None:
        return np.zeros(target_size, dtype=np.float32)
    
    # Quick resize
    zoom_factors = [t/s for t, s in zip(target_size, volume.shape)]
    volume_resized = zoom(volume, zoom_factors, order=1)
    
    return volume_resized.astype(np.float32)

# Process in batches
processed_data = {}
failed = []

print("\n🚀 Starting simplified processing...")
for i in tqdm(range(0, len(train_df), CONFIG['BATCH_SIZE']), desc="Batches"):
    batch_df = train_df.iloc[i:i+CONFIG['BATCH_SIZE']]
    
    for _, row in batch_df.iterrows():
        series_id = row['SeriesInstanceUID']
        series_path = Path(CONFIG['SERIES_DIR']) / series_id
        
        try:
            # Load
            volume, meta = load_dicom_volume(series_path)
            
            if volume is not None:
                # Process
                processed = process_volume_simple(volume)
                
                # Store in memory (or save to disk if needed)
                processed_data[series_id] = {
                    'volume': processed,
                    'labels': row[LABEL_COLS].values.astype(np.float32)
                }
            else:
                failed.append(series_id)
                
        except Exception as e:
            failed.append(series_id)
            continue

print(f"\n✅ Processed: {len(processed_data)}/{len(train_df)}")
print(f"❌ Failed: {len(failed)}")

# Optional: Save to disk as NPZ files for later use
if len(processed_data) > 0:
    import os
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    print("\n💾 Saving to disk...")
    for series_id, data in tqdm(list(processed_data.items())[:100], desc="Saving"):
        np.savez_compressed(
            f"{CONFIG['OUTPUT_DIR']}/{series_id}.npz",
            volume=data['volume'],
            labels=data['labels']
        )
    print("✅ Saved samples to disk")
