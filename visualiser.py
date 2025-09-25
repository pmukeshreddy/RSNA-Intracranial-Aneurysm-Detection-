
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class DICOM3DVisualizer:
    def __init__(self, npz_path):
        """
        Initialize the visualizer with NPZ file path
        """
        self.npz_path = npz_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """
        Load and inspect NPZ file contents
        """
        try:
            npz_file = np.load(self.npz_path)
            print("Available arrays in NPZ file:")
            for key in npz_file.files:
                print(f"- {key}: shape {npz_file[key].shape}")
            
            # Try to find the main image data
            if len(npz_file.files) == 1:
                # Single array case
                key = npz_file.files[0]
                self.data = npz_file[key]
            else:
                # Multiple arrays - look for common DICOM array names
                possible_keys = ['image', 'volume', 'data', 'dicom', 'array']
                found_key = None
                
                for key in npz_file.files:
                    if any(pk in key.lower() for pk in possible_keys):
                        found_key = key
                        break
                
                if found_key:
                    self.data = npz_file[found_key]
                else:
                    # Use the largest 3D array
                    largest_key = max(npz_file.files, 
                                    key=lambda k: np.prod(npz_file[k].shape) if len(npz_file[k].shape) == 3 else 0)
                    self.data = npz_file[largest_key]
            
            print(f"\nLoaded data shape: {self.data.shape}")
            print(f"Data type: {self.data.dtype}")
            print(f"Value range: {self.data.min()} to {self.data.max()}")
            
            # Normalize data if needed
            if self.data.max() > 1.0:
                self.data = self.normalize_data(self.data)
                
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            return None
    
    def normalize_data(self, data):
        """
        Normalize data to 0-1 range
        """
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def slice_viewer_interactive(self, axis=0):
        """
        Interactive slice viewer with slider
        """
        if self.data is None:
            print("No data loaded!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial slice
        slice_idx = self.data.shape[axis] // 2
        
        if axis == 0:
            im = ax.imshow(self.data[slice_idx], cmap='gray', aspect='auto')
            ax.set_title(f'Axial Slice {slice_idx}')
        elif axis == 1:
            im = ax.imshow(self.data[:, slice_idx], cmap='gray', aspect='auto')
            ax.set_title(f'Coronal Slice {slice_idx}')
        else:
            im = ax.imshow(self.data[:, :, slice_idx], cmap='gray', aspect='auto')
            ax.set_title(f'Sagittal Slice {slice_idx}')
        
        ax.axis('off')
        
        # Slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, self.data.shape[axis]-1, 
                       valinit=slice_idx, valfmt='%d')
        
        def update(val):
            idx = int(slider.val)
            if axis == 0:
                im.set_array(self.data[idx])
                ax.set_title(f'Axial Slice {idx}')
            elif axis == 1:
                im.set_array(self.data[:, idx])
                ax.set_title(f'Coronal Slice {idx}')
            else:
                im.set_array(self.data[:, :, idx])
                ax.set_title(f'Sagittal Slice {idx}')
            fig.canvas.draw()
        
        slider.on_changed(update)
        plt.show()
    
    def multi_plane_view(self, slice_indices=None):
        """
        Show three orthogonal views
        """
        if self.data is None:
            print("No data loaded!")
            return
        
        if slice_indices is None:
            slice_indices = [s//2 for s in self.data.shape]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial
        axes[0].imshow(self.data[slice_indices[0]], cmap='gray', aspect='auto')
        axes[0].set_title(f'Axial (slice {slice_indices[0]})')
        axes[0].axis('off')
        
        # Coronal
        axes[1].imshow(self.data[:, slice_indices[1]], cmap='gray', aspect='auto')
        axes[1].set_title(f'Coronal (slice {slice_indices[1]})')
        axes[1].axis('off')
        
        # Sagittal
        axes[2].imshow(self.data[:, :, slice_indices[2]], cmap='gray', aspect='auto')
        axes[2].set_title(f'Sagittal (slice {slice_indices[2]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def volume_render_simple(self, threshold=0.5, alpha=0.1):
        """
        Simple 3D volume rendering using scatter plot
        """
        if self.data is None:
            print("No data loaded!")
            return
        
        # Downsample for performance
        step = max(1, min(self.data.shape) // 50)
        downsampled = self.data[::step, ::step, ::step]
        
        # Find voxels above threshold
        mask = downsampled > threshold
        z, y, x = np.where(mask)
        values = downsampled[mask]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, y, z, c=values, cmap='viridis', 
                           alpha=alpha, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Volume Rendering')
        
        plt.colorbar(scatter)
        plt.show()
    
    def histogram_analysis(self):
        """
        Show intensity histogram
        """
        if self.data is None:
            print("No data loaded!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(self.data.flatten(), bins=100, alpha=0.7, color='blue')
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Intensity Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative histogram
        counts, bins = np.histogram(self.data.flatten(), bins=100)
        ax2.plot(bins[:-1], np.cumsum(counts))
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Cumulative Frequency')
        ax2.set_title('Cumulative Intensity Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def montage_view(self, axis=0, cols=6):
        """
        Show multiple slices in a montage
        """
        if self.data is None:
            print("No data loaded!")
            return
        
        n_slices = self.data.shape[axis]
        step = max(1, n_slices // (cols * cols))
        slice_indices = range(0, n_slices, step)[:cols*cols]
        
        rows = len(slice_indices) // cols + (1 if len(slice_indices) % cols else 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows*2.5))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(slice_indices):
            row, col = i // cols, i % cols
            
            if axis == 0:
                img = self.data[idx]
            elif axis == 1:
                img = self.data[:, idx]
            else:
                img = self.data[:, :, idx]
            
            axes[row, col].imshow(img, cmap='gray', aspect='auto')
            axes[row, col].set_title(f'Slice {idx}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(len(slice_indices), rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    # Initialize visualizer
    npz_file_path = "/content/drive/MyDrive/kaggle_sep_2025_main/New Folder With Items 2/preprocessed_data_part01_of_28/1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317.npz"  # Replace with your file path
    visualizer = DICOM3DVisualizer(npz_file_path)
    
    if visualizer.data is not None:
        # Different visualization options
        
        # 1. Interactive slice viewer (axial)
        print("1. Interactive axial slice viewer")
        visualizer.slice_viewer_interactive(axis=0)
        
        # 2. Multi-plane view
        print("2. Multi-plane view")
        visualizer.multi_plane_view()
        
        # 3. Montage view
        print("3. Montage view")
        visualizer.montage_view(axis=0, cols=6)
        
        # 4. Histogram analysis
        print("4. Histogram analysis")
        visualizer.histogram_analysis()
        
        # 5. Simple 3D volume rendering
        print("5. 3D volume rendering")
        visualizer.volume_render_simple(threshold=0.3, alpha=0.1)

if __name__ == "__main__":
    main()
