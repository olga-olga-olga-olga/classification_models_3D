import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import os

def convert_nifti_to_npy(nifti_file_tuples, output_dir, target_shape=(240, 240, 160)):
    """
    Convert NIfTI files to .npy format that datagen expects
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_paths = []
    
    for i, (t1_path, t2_path, flair_path) in enumerate(nifti_file_tuples):
        # Create patient ID
        patient_id = f"test_patient_{i:04d}"
        
        # Load NIfTI files
        t1_data = nib.load(t1_path).get_fdata()
        t2_data = nib.load(t2_path).get_fdata()
        flair_data = nib.load(flair_path).get_fdata()
        
        # Resample to target shape if needed
        if t1_data.shape != target_shape:
            zoom_factors = [target_shape[i] / t1_data.shape[i] for i in range(3)]
            t1_data = zoom(t1_data, zoom_factors, order=1)
            t2_data = zoom(t2_data, zoom_factors, order=1)
            flair_data = zoom(flair_data, zoom_factors, order=1)
        
        # Stack into 3-channel format (same as training)
        volume = np.stack([t1_data, t2_data, flair_data], axis=-1)
        
        # Apply same normalization as training data
        volume = normalize_intensity(volume)
        
        # Save as .npy file
        output_path = os.path.join(output_dir, f"{patient_id}.npy")
        np.save(output_path, volume.astype(np.float16))
        npy_paths.append(output_path)
    
    return npy_paths

def normalize_intensity(volume):
    """
    Apply intensity normalization - MUST match training data preprocessing
    """
    # You need to determine the exact normalization used in training
    # Common approach: per-channel z-score normalization
    for channel in range(volume.shape[-1]):
        channel_data = volume[..., channel]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            volume[..., channel] = (channel_data - mean) / std
    
    return volume