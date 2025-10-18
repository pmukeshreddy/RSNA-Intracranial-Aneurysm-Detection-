import os
import subprocess
import json
import time
import shutil
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional


class DICOMPreprocessor:
    def __init__(self, dcm2niix_path='dcm2niix'):
        self.dcm2niix_path = dcm2niix_path
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required tools are available."""
        if not shutil.which(self.dcm2niix_path):
            raise RuntimeError(
                f"dcm2niix not found. Install it with: "
                f"'apt-get install dcm2niix' or 'conda install -c conda-forge dcm2niix'"
            )
        
        self.has_gdcm = shutil.which('gdcmconv') is not None
        if not self.has_gdcm:
            print("Warning: gdcmconv not found. Fallback conversion disabled.")
    
    def get_series_info(self, dicom_dir):
        dicom_files = sorted(list(dicom_dir.glob("*.dcm")))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        configs = defaultdict(list)
        spacings = []
        
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                if not hasattr(dcm, 'PixelSpacing'):
                    continue
                rows = int(dcm.Rows)
                cols = int(dcm.Columns)
                pixel_spacing = tuple(float(x) for x in dcm.PixelSpacing)
                config = (rows, cols, pixel_spacing)
                configs[config].append(dcm_file)

                if hasattr(dcm, "SliceThickness"):
                    spacings.append(float(dcm.SliceThickness))
                elif hasattr(dcm, "SpacingBetweenSlices"):
                    spacings.append(float(dcm.SpacingBetweenSlices))
            except Exception as e:
                print(f"Error reading {dcm_file}: {e}")
                continue
        
        return configs, spacings
    
    def filter_outliers(self, dicom_dir):
        configs, spacings = self.get_series_info(dicom_dir)
        if not configs:
            return []

        max_config = max(configs.items(), key=lambda x: len(x[1]))
        valid_files = max_config[1]

        if spacings:
            spacings_array = np.array(spacings)
            q1, q3 = np.percentile(spacings_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered_files = []
            for f in valid_files:
                dcm = pydicom.dcmread(str(f), stop_before_pixels=True)
                spacing = float(dcm.get("SliceThickness", dcm.get("SpacingBetweenSlices", 0)))
                if lower_bound <= spacing <= upper_bound:
                    filtered_files.append(f)
            valid_files = filtered_files
        
        return valid_files

    def convert_to_nifti(self, dicom_dir: Path, output_dir: Path, 
                        use_gdcm_fallback: bool = True) -> Optional[Path]:
        """
        Convert DICOM series to NIfTI using dcm2niix.
        Falls back to gdcmconv if initial conversion fails.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter to valid DICOM files
        valid_files = self.filter_outliers(dicom_dir)
        
        if len(valid_files) < 3:
            print(f"Insufficient valid files in {dicom_dir}")
            return None
        
        # Create temp directory with filtered files
        temp_dir = output_dir / "temp_filtered"
        temp_dir.mkdir(exist_ok=True)
        
        for i, f in enumerate(valid_files):
            os.symlink(f, temp_dir / f"{i:04d}.dcm")
        
        try:
            # Try dcm2niix conversion
            cmd = [
                self.dcm2niix_path,
                "-b", "y",  # Output JSON sidecar
                "-i", "y",  # Ignore derived/localizer
                "-z", "y",  # Compress output
                "-f", dicom_dir.name,  # Output filename
                "-o", str(output_dir),
                str(temp_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0 and use_gdcm_fallback and self.has_gdcm:
                print(f"dcm2niix failed, trying gdcmconv fallback...")
                return self._gdcm_fallback(temp_dir, output_dir, dicom_dir.name)
            
            # Find output file
            nifti_files = list(output_dir.glob(f"{dicom_dir.name}*.nii.gz"))
            if nifti_files:
                return nifti_files[0]
                
        except Exception as e:
            print(f"Conversion error: {e}")
            if use_gdcm_fallback and self.has_gdcm:
                return self._gdcm_fallback(temp_dir, output_dir, dicom_dir.name)
        
        finally:
            # Cleanup temp directory
            for f in temp_dir.glob("*.dcm"):
                f.unlink()
            temp_dir.rmdir()
        
        return None

    def _gdcm_fallback(self, dicom_dir: Path, output_dir: Path, 
                       series_name: str) -> Optional[Path]:
        """Fallback using gdcmconv to clean DICOM before conversion."""
        gdcm_dir = output_dir / "gdcm_temp"
        gdcm_dir.mkdir(exist_ok=True)
        
        # Convert with gdcmconv
        for dcm_file in dicom_dir.glob("*.dcm"):
            output_file = gdcm_dir / dcm_file.name
            cmd = ["gdcmconv", "--raw", str(dcm_file), str(output_file)]
            subprocess.run(cmd, capture_output=True)
        
        # Try dcm2niix again
        cmd = [
            self.dcm2niix_path,
            "-b", "y",
            "-i", "y",
            "-z", "y",
            "-f", series_name,
            "-o", str(output_dir),
            str(gdcm_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Cleanup
        for f in gdcm_dir.glob("*"):
            f.unlink()
        gdcm_dir.rmdir()
        
        if result.returncode == 0:
            nifti_files = list(output_dir.glob(f"{series_name}*.nii.gz"))
            if nifti_files:
                return nifti_files[0]
        
        return None

    def process_dataset(self, input_root: Path, output_root: Path, 
                   exclude_list: Optional[List[str]] = None,
                   chunk_id: int = 0, 
                   total_chunks: int = 1):
        exclude_list = exclude_list or []
        
        # Get all series directories
        series_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
        series_dirs = [d for d in series_dirs if d.name not in exclude_list]
        
        # Split into chunks
        chunk_size = len(series_dirs) // total_chunks
        start_idx = chunk_id * chunk_size
        end_idx = start_idx + chunk_size if chunk_id < total_chunks - 1 else len(series_dirs)
        series_dirs = series_dirs[start_idx:end_idx]
        
        total = len(series_dirs)
        print(f"Processing chunk {chunk_id + 1}/{total_chunks}: {total} series")
        
        successful = []
        failed = []
        
        start_time = time.time()
        
        for idx, series_dir in enumerate(series_dirs, 1):
            series_id = series_dir.name
            
            if series_id in exclude_list:
                print(f"Skipping excluded series: {series_id}")
                continue
            
            print(f"Processing {series_id}...")
            output_dir = output_root / series_id
            
            result = self.convert_to_nifti(series_dir, output_dir)
            
            if result:
                successful.append(series_id)
                status = "✓ Success"
            else:
                failed.append(series_id)
                status = "✗ Failed"
            
            # Progress and ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (total - idx) * avg_time
            
            print(f"[{idx}/{total}] {status}: {series_id}")
            print(f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m\n")
        
        print(f"\nProcessing complete:")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        return successful, failed
