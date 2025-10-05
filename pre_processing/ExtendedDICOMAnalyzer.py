import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ExtendedDICOMAnalyzer:
    def __init__(self, data_dir, train_csv_path):
        self.data_dir = Path(data_dir)
        self.train_csv_path = train_csv_path
        self.series_info = defaultdict(list)
        self.labels_df = None
        
    def analyze_series_structure(self):
        """Analyze series-level structure: slices, spacing, etc."""
        print("\n🔬 ANALYZING SERIES STRUCTURE...")
        print("-" * 60)
        
        all_files = list(self.data_dir.rglob('*.dcm'))
        
        # Group files by SeriesInstanceUID
        series_groups = defaultdict(list)
        
        print("📂 Grouping files by series...")
        for file_path in tqdm(all_files, desc="Reading series info"):
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                series_uid = dcm.SeriesInstanceUID
                
                series_groups[series_uid].append({
                    'path': str(file_path),
                    'instance_num': int(getattr(dcm, 'InstanceNumber', 0)),
                    'slice_location': float(getattr(dcm, 'SliceLocation', 0.0)),
                    'slice_thickness': float(getattr(dcm, 'SliceThickness', 0.0)),
                    'pixel_spacing': getattr(dcm, 'PixelSpacing', [0, 0]),
                    'modality': getattr(dcm, 'Modality', 'Unknown')
                })
            except Exception as e:
                continue
        
        # Analyze each series
        series_stats = []
        
        print(f"\n📊 Analyzing {len(series_groups)} unique series...")
        for series_uid, slices in tqdm(series_groups.items(), desc="Computing stats"):
            slices_sorted = sorted(slices, key=lambda x: x['slice_location'])
            
            num_slices = len(slices_sorted)
            modality = slices_sorted[0]['modality']
            
            # Calculate slice spacing
            if num_slices > 1:
                locations = [s['slice_location'] for s in slices_sorted]
                spacings = np.diff(locations)
                avg_spacing = np.mean(np.abs(spacings))
                spacing_std = np.std(np.abs(spacings))
                max_gap = np.max(np.abs(spacings))
            else:
                avg_spacing = 0
                spacing_std = 0
                max_gap = 0
            
            slice_thickness = slices_sorted[0]['slice_thickness']
            
            try:
                pixel_spacing = slices_sorted[0]['pixel_spacing']
                if isinstance(pixel_spacing, list):
                    pixel_spacing_val = float(pixel_spacing[0])
                else:
                    pixel_spacing_val = float(pixel_spacing)
            except:
                pixel_spacing_val = 0.0
            
            series_stats.append({
                'SeriesInstanceUID': series_uid,
                'modality': modality,
                'num_slices': num_slices,
                'avg_slice_spacing': avg_spacing,
                'spacing_std': spacing_std,
                'max_spacing_gap': max_gap,
                'slice_thickness': slice_thickness,
                'pixel_spacing': pixel_spacing_val
            })
        
        self.series_df = pd.DataFrame(series_stats)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SERIES STRUCTURE SUMMARY")
        print("="*60)
        
        print(f"\n📍 Total unique series: {len(self.series_df)}")
        
        for modality in self.series_df['modality'].unique():
            mod_data = self.series_df[self.series_df['modality'] == modality]
            
            print(f"\n📌 {modality} (n={len(mod_data)} series):")
            print(f"  Slices per series:")
            print(f"    Min  : {mod_data['num_slices'].min()}")
            print(f"    Max  : {mod_data['num_slices'].max()}")
            print(f"    Mean : {mod_data['num_slices'].mean():.1f}")
            print(f"    Median: {mod_data['num_slices'].median():.0f}")
            
            print(f"  Slice spacing (mm):")
            print(f"    Mean : {mod_data['avg_slice_spacing'].mean():.2f}")
            print(f"    Std  : {mod_data['spacing_std'].mean():.2f}")
            print(f"    Max gap: {mod_data['max_spacing_gap'].mean():.2f}")
            
            print(f"  Slice thickness (mm):")
            print(f"    Range: [{mod_data['slice_thickness'].min():.1f}, "
                  f"{mod_data['slice_thickness'].max():.1f}]")
            print(f"    Mean : {mod_data['slice_thickness'].mean():.2f}")
            
            # Check for problematic series
            too_few_slices = len(mod_data[mod_data['num_slices'] < 3])
            irregular_spacing = len(mod_data[mod_data['spacing_std'] > 1.0])
            
            if too_few_slices > 0:
                print(f"  ⚠️  {too_few_slices} series with <3 slices (can't use 2.5D)")
            if irregular_spacing > 0:
                print(f"  ⚠️  {irregular_spacing} series with irregular spacing")
        
        return self.series_df
    
    def analyze_labels(self):
        """Analyze label distribution and class imbalance"""
        print("\n\n🏷️ ANALYZING LABEL DISTRIBUTION...")
        print("-" * 60)
        
        self.labels_df = pd.read_csv(self.train_csv_path)
        
        # Get label columns
        meta_cols = ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality']
        label_cols = [col for col in self.labels_df.columns if col not in meta_cols]
        
        print(f"\n📊 Label columns: {len(label_cols)}")
        print(f"📊 Total samples: {len(self.labels_df)}")
        
        # Merge with series info if available
        if hasattr(self, 'series_df'):
            merged = self.labels_df.merge(
                self.series_df[['SeriesInstanceUID', 'modality', 'num_slices']], 
                on='SeriesInstanceUID', 
                how='left'
            )
        else:
            merged = self.labels_df
        
        print("\n" + "="*60)
        print("📊 CLASS DISTRIBUTION (Multi-Label)")
        print("="*60)
        
        # Per-class statistics
        label_stats = []
        for col in tqdm(label_cols, desc="📈 Computing class stats"):
            positive = merged[col].sum()
            negative = len(merged) - positive
            pos_ratio = positive / len(merged) * 100
            
            label_stats.append({
                'label': col,
                'positive': positive,
                'negative': negative,
                'pos_ratio': pos_ratio
            })
        
        label_stats_df = pd.DataFrame(label_stats).sort_values('pos_ratio', ascending=False)
        
        print("\n📊 Per-Class Breakdown:")
        for _, row in label_stats_df.iterrows():
            bar_length = int(row['pos_ratio'] / 2)
            bar = '█' * bar_length
            print(f"  {row['label']:30s}: {row['positive']:5.0f} / {len(merged):5d} "
                  f"({row['pos_ratio']:5.1f}%) {bar}")
        
        # Check for severe imbalance
        severe_imbalance = label_stats_df[label_stats_df['pos_ratio'] < 5]
        if len(severe_imbalance) > 0:
            print(f"\n⚠️  {len(severe_imbalance)} classes with <5% positive samples:")
            for _, row in severe_imbalance.iterrows():
                print(f"    {row['label']:30s}: {row['pos_ratio']:.2f}%")
        
        # Multi-label co-occurrence
        print("\n📊 Multi-Label Statistics:")
        labels_per_sample = merged[label_cols].sum(axis=1)
        print(f"  Samples with 0 labels: {(labels_per_sample == 0).sum()}")
        print(f"  Samples with 1 label : {(labels_per_sample == 1).sum()}")
        print(f"  Samples with 2+ labels: {(labels_per_sample >= 2).sum()}")
        print(f"  Max labels per sample: {labels_per_sample.max()}")
        print(f"  Avg labels per sample: {labels_per_sample.mean():.2f}")
        
        # Modality-specific label distribution
        if 'modality' in merged.columns:
            print("\n📊 Label Distribution by Modality:")
            for modality in merged['modality'].unique():
                mod_data = merged[merged['modality'] == modality]
                mod_labels = mod_data[label_cols].sum(axis=1)
                print(f"\n  {modality}:")
                print(f"    Total samples: {len(mod_data)}")
                print(f"    Avg labels/sample: {mod_labels.mean():.2f}")
                print(f"    Positive samples: {(mod_labels > 0).sum()} "
                      f"({(mod_labels > 0).sum() / len(mod_data) * 100:.1f}%)")
        
        self.label_stats_df = label_stats_df
        return label_stats_df
    
    def analyze_patient_level(self):
        """Patient-level analysis for data leakage prevention"""
        print("\n\n👤 PATIENT-LEVEL ANALYSIS...")
        print("-" * 60)
        
        if self.labels_df is None:
            self.labels_df = pd.read_csv(self.train_csv_path)
        
        # Extract patient ID from SeriesInstanceUID (pattern varies)
        # Try common patterns
        print("🔍 Extracting patient identifiers...")
        
        # Method 1: Check if there's a pattern in SeriesInstanceUID
        sample_series = self.labels_df['SeriesInstanceUID'].head(100).tolist()
        
        # Simple approach: use first part before first number change
        patient_ids = []
        for series_uid in tqdm(self.labels_df['SeriesInstanceUID'], desc="Extracting patient IDs"):
            # Use the series UID prefix as patient identifier
            parts = series_uid.split('.')
            if len(parts) >= 3:
                patient_id = '.'.join(parts[:3])  # Use first 3 parts
            else:
                patient_id = series_uid
            patient_ids.append(patient_id)
        
        self.labels_df['patient_id'] = patient_ids
        
        print("\n" + "="*60)
        print("📊 PATIENT-LEVEL SUMMARY")
        print("="*60)
        
        unique_patients = self.labels_df['patient_id'].nunique()
        total_series = len(self.labels_df)
        
        print(f"\n📍 Unique patients: {unique_patients}")
        print(f"📍 Total series: {total_series}")
        print(f"📍 Avg series per patient: {total_series / unique_patients:.2f}")
        
        # Series per patient distribution
        series_per_patient = self.labels_df.groupby('patient_id').size()
        print(f"\n📊 Series per patient:")
        print(f"  Min  : {series_per_patient.min()}")
        print(f"  Max  : {series_per_patient.max()}")
        print(f"  Mean : {series_per_patient.mean():.2f}")
        print(f"  Median: {series_per_patient.median():.0f}")
        
        # Patients with multiple series
        multi_series = (series_per_patient > 1).sum()
        print(f"\n⚠️  Patients with multiple series: {multi_series} "
              f"({multi_series / unique_patients * 100:.1f}%)")
        print(f"   → MUST split by patient, not series, to avoid data leakage!")
        
        # Age/Sex analysis if available
        if 'PatientAge' in self.labels_df.columns:
            print(f"\n📊 Patient Age:")
            print(f"  Range: [{self.labels_df['PatientAge'].min()}, "
                  f"{self.labels_df['PatientAge'].max()}]")
            print(f"  Mean : {self.labels_df['PatientAge'].mean():.1f}")
            print(f"  Median: {self.labels_df['PatientAge'].median():.0f}")
        
        if 'PatientSex' in self.labels_df.columns:
            print(f"\n📊 Patient Sex:")
            sex_counts = self.labels_df['PatientSex'].value_counts()
            for sex, count in sex_counts.items():
                print(f"  {sex}: {count} ({count / len(self.labels_df) * 100:.1f}%)")
        
        return series_per_patient
    
    def clean_window_settings(self):
        """Fix window settings string/float issues"""
        print("\n\n🔧 CLEANING WINDOW SETTINGS...")
        print("-" * 60)
        
        cleaned_windows = defaultdict(lambda: defaultdict(list))
        
        all_files = list(self.data_dir.rglob('*.dcm'))
        sample_files = np.random.choice(all_files, min(1000, len(all_files)), replace=False)
        
        for file_path in tqdm(sample_files, desc="Extracting clean windows"):
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                modality = getattr(dcm, 'Modality', 'Unknown')
                
                wc = getattr(dcm, 'WindowCenter', None)
                ww = getattr(dcm, 'WindowWidth', None)
                
                if wc is not None and ww is not None:
                    # Convert to list if needed
                    if not isinstance(wc, (list, pydicom.multival.MultiValue)):
                        wc = [wc]
                    if not isinstance(ww, (list, pydicom.multival.MultiValue)):
                        ww = [ww]
                    
                    # Convert all to float
                    wc_clean = [float(x) for x in wc]
                    ww_clean = [float(x) for x in ww]
                    
                    for c, w in zip(wc_clean, ww_clean):
                        cleaned_windows[modality]['centers'].append(c)
                        cleaned_windows[modality]['widths'].append(w)
            
            except Exception as e:
                continue
        
        print("\n" + "="*60)
        print("🪟 CLEANED WINDOW RECOMMENDATIONS")
        print("="*60)
        
        recommended_params = {}
        
        for modality in cleaned_windows:
            centers = np.array(cleaned_windows[modality]['centers'])
            widths = np.array(cleaned_windows[modality]['widths'])
            
            print(f"\n📌 {modality}:")
            
            # Get most common window pairs
            window_pairs = list(zip(centers, widths))
            most_common = Counter(window_pairs).most_common(5)
            
            print(f"  Most common windows:")
            for (c, w), count in most_common:
                print(f"    Center={c:.0f}, Width={w:.0f} → used {count} times")
            
            # Recommend based on median
            median_centers = np.median(centers)
            median_widths = np.median(widths)
            
            print(f"\n  Recommended (median):")
            print(f"    Center: {median_centers:.0f}")
            print(f"    Width : {median_widths:.0f}")
            
            recommended_params[modality] = {
                'center': median_centers,
                'width': median_widths,
                'most_common': most_common[0] if most_common else None
            }
        
        return recommended_params
    
    def plot_comprehensive_analysis(self, save_dir='/kaggle/working'):
        """Generate comprehensive plots"""
        save_dir = Path(save_dir)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Series slice count distribution
        if hasattr(self, 'series_df'):
            ax1 = plt.subplot(2, 3, 1)
            for modality in self.series_df['modality'].unique():
                data = self.series_df[self.series_df['modality'] == modality]['num_slices']
                ax1.hist(data, bins=50, alpha=0.6, label=modality)
            ax1.set_xlabel('Number of Slices per Series')
            ax1.set_ylabel('Count')
            ax1.set_title('Series Slice Count Distribution')
            ax1.legend()
            ax1.axvline(3, color='red', linestyle='--', label='Min for 2.5D')
        
        # 2. Slice spacing distribution
        if hasattr(self, 'series_df'):
            ax2 = plt.subplot(2, 3, 2)
            for modality in self.series_df['modality'].unique():
                data = self.series_df[self.series_df['modality'] == modality]['avg_slice_spacing']
                ax2.hist(data[data > 0], bins=50, alpha=0.6, label=modality)
            ax2.set_xlabel('Average Slice Spacing (mm)')
            ax2.set_ylabel('Count')
            ax2.set_title('Slice Spacing Distribution')
            ax2.legend()
        
        # 3. Label distribution
        if hasattr(self, 'label_stats_df'):
            ax3 = plt.subplot(2, 3, 3)
            top_labels = self.label_stats_df.head(10)
            ax3.barh(range(len(top_labels)), top_labels['pos_ratio'])
            ax3.set_yticks(range(len(top_labels)))
            ax3.set_yticklabels(top_labels['label'])
            ax3.set_xlabel('Positive Ratio (%)')
            ax3.set_title('Top 10 Label Distribution')
        
        # 4. Multi-label co-occurrence
        if self.labels_df is not None:
            ax4 = plt.subplot(2, 3, 4)
            meta_cols = ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'patient_id']
            label_cols = [col for col in self.labels_df.columns if col not in meta_cols]
            labels_per_sample = self.labels_df[label_cols].sum(axis=1)
            ax4.hist(labels_per_sample, bins=range(0, labels_per_sample.max() + 2), edgecolor='black')
            ax4.set_xlabel('Number of Labels per Sample')
            ax4.set_ylabel('Count')
            ax4.set_title('Multi-Label Distribution')
        
        # 5. Patient series distribution
        if 'patient_id' in self.labels_df.columns:
            ax5 = plt.subplot(2, 3, 5)
            series_per_patient = self.labels_df.groupby('patient_id').size()
            ax5.hist(series_per_patient, bins=50, edgecolor='black')
            ax5.set_xlabel('Series per Patient')
            ax5.set_ylabel('Count')
            ax5.set_title('Patient Series Distribution')
            ax5.axvline(1, color='red', linestyle='--', alpha=0.5)
        
        # 6. Age distribution (if available)
        if 'PatientAge' in self.labels_df.columns:
            ax6 = plt.subplot(2, 3, 6)
            ax6.hist(self.labels_df['PatientAge'].dropna(), bins=30, edgecolor='black')
            ax6.set_xlabel('Patient Age')
            ax6.set_ylabel('Count')
            ax6.set_title('Age Distribution')
        
        plt.tight_layout()
        plot_path = save_dir / 'comprehensive_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Comprehensive plots saved to {plot_path}")
        plt.show()
    
    def generate_preprocessing_config(self, output_path='/kaggle/working/preprocessing_config.json'):
        """Generate final preprocessing configuration"""
        print("\n\n⚙️ GENERATING PREPROCESSING CONFIG...")
        print("-" * 60)
        
        config = {
            'series_stats': {},
            'recommended_windows': {},
            'warnings': []
        }
        
        # Series recommendations
        if hasattr(self, 'series_df'):
            for modality in self.series_df['modality'].unique():
                mod_data = self.series_df[self.series_df['modality'] == modality]
                
                min_slices = int(mod_data['num_slices'].min())
                median_slices = int(mod_data['num_slices'].median())
                
                config['series_stats'][modality] = {
                    'min_slices': min_slices,
                    'median_slices': median_slices,
                    'avg_spacing_mm': float(mod_data['avg_slice_spacing'].mean()),
                    'recommended_slice_context': 3 if min_slices >= 3 else 1
                }
                
                if min_slices < 3:
                    config['warnings'].append(
                        f"{modality}: Some series have <3 slices, 2.5D may not work for all"
                    )
        
        # Save
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config saved to {output_path}")
        print("\nKey findings:")
        print(json.dumps(config, indent=2))
        
        return config


# ==================== USAGE ====================
if __name__ == "__main__":
    DATA_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection/train'
    TRAIN_CSV = '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'
    
    analyzer = ExtendedDICOMAnalyzer(DATA_DIR, TRAIN_CSV)
    
    # Run all analyses
    print("🚀 Running Extended Analysis Pipeline...\n")
    
    # 1. Series structure
    series_df = analyzer.analyze_series_structure()
    
    # 2. Label distribution
    label_stats = analyzer.analyze_labels()
    
    # 3. Patient-level
    patient_stats = analyzer.analyze_patient_level()
    
    # 4. Clean windows
    clean_windows = analyzer.clean_window_settings()
    
    # 5. Generate plots
    analyzer.plot_comprehensive_analysis()
    
    # 6. Generate config
    config = analyzer.generate_preprocessing_config()
    
    # Save results
    series_df.to_csv('/kaggle/working/series_analysis.csv', index=False)
    label_stats.to_csv('/kaggle/working/label_analysis.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ EXTENDED ANALYSIS COMPLETE!")
    print("="*60)
    print("\nFiles saved:")
    print("  - /kaggle/working/series_analysis.csv")
    print("  - /kaggle/working/label_analysis.csv")
    print("  - /kaggle/working/preprocessing_config.json")
    print("  - /kaggle/working/comprehensive_analysis.png")
