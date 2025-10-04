# DEBUG SAMPLE 3 + FULL SCAN WITH PROGRESS BAR
from tqdm import tqdm

print("="*70)
print("DEBUGGING SAMPLE 3 (ALL ZEROS)")
print("="*70)

# Get the series ID for sample 3
series_id = dataset_no_aug.df.iloc[3]['SeriesInstanceUID']
npy_path = dataset_no_aug.file_mapping.get(series_id)

print(f"Series ID: {series_id}")
print(f"File path: {npy_path}")
print(f"File exists: {npy_path.exists() if npy_path else 'N/A'}")

if npy_path and npy_path.exists():
    # Load raw file
    raw_data = np.load(npy_path, allow_pickle=True).astype(np.float32)
    print(f"\nRaw file info:")
    print(f"  Shape: {raw_data.shape}")
    print(f"  Min: {raw_data.min()}")
    print(f"  Max: {raw_data.max()}")
    print(f"  Mean: {raw_data.mean()}")
    print(f"  All zeros: {(raw_data == 0).all()}")
    
    # If 4D, take first channel
    if raw_data.ndim == 4:
        raw_data = raw_data[0]
        print(f"\nAfter taking first channel:")
        print(f"  Shape: {raw_data.shape}")
        print(f"  Min: {raw_data.min()}")
        print(f"  Max: {raw_data.max()}")
    
    # Check preprocessing condition
    is_normalized = (raw_data.max() < 2.0 and raw_data.min() >= -0.1)
    print(f"\nPreprocessing check:")
    print(f"  max < 2.0: {raw_data.max()} < 2.0 = {raw_data.max() < 2.0}")
    print(f"  min >= -0.1: {raw_data.min()} >= -0.1 = {raw_data.min() >= -0.1}")
    print(f"  Will use NORMALIZED path: {is_normalized}")
    
    if (raw_data == 0).all():
        print("\n❌ Source file is completely empty/zeros - CORRUPTED FILE")
    else:
        print("\n⚠️ File has data but becomes zeros after processing")
        print("   The _resize_volume() method might be the issue")

# ============================================
# SCAN WITH PROGRESS BAR
# ============================================
print("\n" + "="*70)
print("SCANNING ENTIRE DATASET FOR BROKEN FILES")
print("="*70)

broken_count = 0
broken_ids = []
broken_details = []

# Use tqdm for progress bar
for i in tqdm(range(len(dataset_no_aug)), desc="Checking samples", unit="sample"):
    X, Y = dataset_no_aug[i]
    if X.max() < 0.01:  # All zeros or near-zeros
        broken_count += 1
        broken_ids.append(i)
        series_id = dataset_no_aug.df.iloc[i]['SeriesInstanceUID']
        broken_details.append((i, series_id, X.max().item()))

print(f"\nTotal samples: {len(dataset_no_aug)}")
print(f"Broken samples: {broken_count}")
print(f"Success rate: {(len(dataset_no_aug) - broken_count) / len(dataset_no_aug) * 100:.1f}%")

if broken_count > 0:
    print(f"\nBroken samples (first 10):")
    for idx, series_id, max_val in broken_details[:10]:
        print(f"  Index {idx}: {series_id} (max={max_val:.6f})")
    
    if broken_count / len(dataset_no_aug) < 0.05:  # Less than 5% broken
        print("\n✓ Acceptable - less than 5% corrupted files")
        print("  You can continue training (these samples will just output zeros)")
    else:
        print(f"\n❌ Too many broken files ({broken_count}/{len(dataset_no_aug)})")
        print("  Need to investigate the preprocessing pipeline")
