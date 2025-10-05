import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class DICOMAnalyzer:
    def __init__(self, data_dir, sample_size=2000):
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.stats = defaultdict(list)
        
    def sample_files(self):
        """Randomly sample DICOM files"""
        print("🔍 Searching for DICOM files...")
        all_files = list(self.data_dir.rglob('*.dcm'))
        
        if len(all_files) > self.sample_size:
            print(f"🎲 Sampling {self.sample_size} from {len(all_files)} total files...")
            sampled = np.random.choice(all_files, self.sample_size, replace=False)
        else:
            sampled = all_files
            print(f"📁 Using all {len(all_files)} files (less than sample size)")
        
        print(f"✅ Selected {len(sampled)} files for analysis\n")
        
        return sampled
    
    def extract_metadata(self, dcm_path):
        """Extract all relevant metadata from DICOM"""
        try:
            dcm = pydicom.dcmread(dcm_path)
            
            # Basic info
            modality = getattr(dcm, 'Modality', 'Unknown')
            manufacturer = getattr(dcm, 'Manufacturer', 'Unknown')
            
            # Image properties
            img = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            
            # Apply rescale
            img_rescaled = img * slope + intercept
            
            # Window settings (if exist)
            window_center = getattr(dcm, 'WindowCenter', None)
            window_width = getattr(dcm, 'WindowWidth', None)
            
            # Handle multiple windows
            if window_center is not None:
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = list(window_center)
                else:
                    window_center = [float(window_center)]
            
            if window_width is not None:
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = list(window_width)
                else:
                    window_width = [float(window_width)]
            
            # Intensity statistics
            stats = {
                'modality': modality,
                'manufacturer': manufacturer,
                'slope': slope,
                'intercept': intercept,
                'pixel_min': float(img.min()),
                'pixel_max': float(img.max()),
                'rescaled_min': float(img_rescaled.min()),
                'rescaled_max': float(img_rescaled.max()),
                'rescaled_mean': float(img_rescaled.mean()),
                'rescaled_std': float(img_rescaled.std()),
                'rescaled_p01': float(np.percentile(img_rescaled, 1)),
                'rescaled_p05': float(np.percentile(img_rescaled, 5)),
                'rescaled_p50': float(np.percentile(img_rescaled, 50)),
                'rescaled_p95': float(np.percentile(img_rescaled, 95)),
                'rescaled_p99': float(np.percentile(img_rescaled, 99)),
                'window_center': window_center,
                'window_width': window_width,
                'bits_stored': getattr(dcm, 'BitsStored', None),
                'photometric': getattr(dcm, 'PhotometricInterpretation', None),
                'rows': img.shape[0],
                'cols': img.shape[1]
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error reading {dcm_path}: {e}")
            return None
    
    def analyze(self):
        """Run full analysis"""
        print("🔍 Starting DICOM Analysis...\n")
        
        sampled_files = self.sample_files()
        
        results = []
        pbar = tqdm(sampled_files, desc="📊 Analyzing DICOM files", 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        for file_path in pbar:
            stats = self.extract_metadata(file_path)
            if stats:
                results.append(stats)
                # Update progress with current modality
                pbar.set_postfix({'Modality': stats['modality'], 'Valid': len(results)})
        
        print(f"\n✅ Successfully analyzed {len(results)}/{len(sampled_files)} files")
        
        self.df = pd.DataFrame(results)
        return self.df
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("📊 DICOM DATASET ANALYSIS REPORT")
        print("="*60)
        
        # Modality distribution
        print("\n1️⃣ MODALITY DISTRIBUTION")
        print("-" * 60)
        modality_counts = self.df['modality'].value_counts()
        for mod, count in modality_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"  {mod:15s}: {count:5d} ({pct:5.1f}%)")
        
        # Manufacturer distribution
        print("\n2️⃣ MANUFACTURER DISTRIBUTION")
        print("-" * 60)
        mfr_counts = self.df['manufacturer'].value_counts().head(5)
        for mfr, count in mfr_counts.items():
            print(f"  {mfr[:30]:30s}: {count:5d}")
        
        # Per-modality statistics
        print("\n3️⃣ INTENSITY STATISTICS PER MODALITY")
        print("-" * 60)
        
        modalities = self.df['modality'].unique()
        for modality in tqdm(modalities, desc="📈 Computing statistics", leave=False):
            mod_data = self.df[self.df['modality'] == modality]
            
            print(f"\n📌 {modality} (n={len(mod_data)})")
            print(f"  Rescaled Range  : [{mod_data['rescaled_min'].min():.1f}, "
                  f"{mod_data['rescaled_max'].max():.1f}]")
            print(f"  Mean ± Std      : {mod_data['rescaled_mean'].mean():.1f} ± "
                  f"{mod_data['rescaled_std'].mean():.1f}")
            print(f"  Percentiles (1-99%): [{mod_data['rescaled_p01'].median():.1f}, "
                  f"{mod_data['rescaled_p99'].median():.1f}]")
            print(f"  Percentiles (5-95%): [{mod_data['rescaled_p05'].median():.1f}, "
                  f"{mod_data['rescaled_p95'].median():.1f}]")
        
        # Window settings analysis
        print("\n4️⃣ DICOM WINDOW SETTINGS")
        print("-" * 60)
        
        for modality in tqdm(modalities, desc="🪟 Analyzing windows", leave=False):
            mod_data = self.df[self.df['modality'] == modality]
            
            # Extract windows that exist
            windows_with_settings = mod_data[mod_data['window_center'].notna()]
            
            if len(windows_with_settings) > 0:
                print(f"\n📌 {modality}: {len(windows_with_settings)}/{len(mod_data)} "
                      f"files have window settings")
                
                # Collect all windows
                all_centers = []
                all_widths = []
                
                for _, row in windows_with_settings.iterrows():
                    if row['window_center']:
                        all_centers.extend(row['window_center'])
                    if row['window_width']:
                        all_widths.extend(row['window_width'])
                
                if all_centers:
                    print(f"  Window Centers  : {set(all_centers)}")
                    print(f"  Window Widths   : {set(all_widths)}")
                    
                    # Most common windows
                    window_pairs = list(zip(
                        windows_with_settings['window_center'].apply(lambda x: tuple(x) if x else None),
                        windows_with_settings['window_width'].apply(lambda x: tuple(x) if x else None)
                    ))
                    window_pairs = [p for p in window_pairs if p[0] is not None]
                    
                    if window_pairs:
                        from collections import Counter
                        common_windows = Counter(window_pairs).most_common(3)
                        print(f"  Most common window combos:")
                        for (centers, widths), count in common_windows:
                            print(f"    Centers={centers}, Widths={widths} → {count} times")
            else:
                print(f"\n📌 {modality}: No predefined window settings found")
        
        # Image dimensions
        print("\n5️⃣ IMAGE DIMENSIONS")
        print("-" * 60)
        print(f"  Rows: min={self.df['rows'].min()}, max={self.df['rows'].max()}, "
              f"median={self.df['rows'].median()}")
        print(f"  Cols: min={self.df['cols'].min()}, max={self.df['cols'].max()}, "
              f"median={self.df['cols'].median()}")
        
        # Recommendations
        print("\n6️⃣ PREPROCESSING RECOMMENDATIONS")
        print("-" * 60)
        
        for modality in self.df['modality'].unique():
            mod_data = self.df[self.df['modality'] == modality]
            
            print(f"\n📌 {modality}:")
            
            if modality == 'CTA':
                # Check if we have predefined windows
                windows_exist = mod_data[mod_data['window_center'].notna()]
                
                if len(windows_exist) > 0:
                    print("  ✅ Use DICOM WindowCenter/Width from files")
                    print("  ⚠️  Fallback if missing:")
                else:
                    print("  ⚠️  No predefined windows, recommend:")
                
                p05 = mod_data['rescaled_p05'].median()
                p95 = mod_data['rescaled_p95'].median()
                p99 = mod_data['rescaled_p99'].median()
                
                # Suggest windows based on actual data
                brain_center = (p05 + p95) / 2
                brain_width = p95 - p05
                
                print(f"    Brain window : center={brain_center:.0f}, width={brain_width:.0f}")
                print(f"    Clip range   : [{mod_data['rescaled_p01'].median():.0f}, "
                      f"{mod_data['rescaled_p99'].median():.0f}]")
            
            elif modality == 'MRA':
                p01 = mod_data['rescaled_p01'].median()
                p99 = mod_data['rescaled_p99'].median()
                
                print(f"  ✅ Use percentile normalization:")
                print(f"    Clip to (1%, 99%): [{p01:.1f}, {p99:.1f}]")
                print(f"    Then normalize to [0, 255]")
        
        print("\n" + "="*60)
    
    def plot_distributions(self, save_path=None):
        """Plot intensity distributions per modality"""
        modalities = self.df['modality'].unique()
        
        fig, axes = plt.subplots(len(modalities), 2, figsize=(15, 5*len(modalities)))
        
        if len(modalities) == 1:
            axes = axes.reshape(1, -1)
        
        print("\n📊 Generating distribution plots...")
        for idx, modality in enumerate(tqdm(modalities, desc="🎨 Creating plots")):
            mod_data = self.df[self.df['modality'] == modality]
            
            # Rescaled min/max distribution
            axes[idx, 0].hist(mod_data['rescaled_min'], bins=50, alpha=0.5, label='Min')
            axes[idx, 0].hist(mod_data['rescaled_max'], bins=50, alpha=0.5, label='Max')
            axes[idx, 0].set_title(f'{modality} - Min/Max Distribution')
            axes[idx, 0].set_xlabel('Intensity Value')
            axes[idx, 0].legend()
            
            # Percentile ranges
            axes[idx, 1].hist(mod_data['rescaled_p01'], bins=30, alpha=0.6, label='p01')
            axes[idx, 1].hist(mod_data['rescaled_p99'], bins=30, alpha=0.6, label='p99')
            axes[idx, 1].set_title(f'{modality} - Percentile Distribution')
            axes[idx, 1].set_xlabel('Intensity Value')
            axes[idx, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_path):
        """Save detailed results to CSV"""
        print("\n💾 Saving detailed results...")
        self.df.to_csv(output_path, index=False)
        print(f"✅ Detailed results saved to {output_path}")


# ==================== USAGE ====================
if __name__ == "__main__":
    DATA_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
    
    analyzer = DICOMAnalyzer(DATA_DIR, sample_size=1000)
    
    # Run analysis
    df_results = analyzer.analyze()
    
    # Generate report
    analyzer.generate_report()
    
    # Plot distributions
    analyzer.plot_distributions(save_path='/kaggle/working/intensity_distributions.png')
    
    # Save detailed results
    analyzer.save_results('/kaggle/working/dicom_analysis_results.csv')
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Review the recommendations above")
    print("2. Update modality_params in preprocessing pipeline")
    print("3. Run preprocessing with data-driven parameters")
