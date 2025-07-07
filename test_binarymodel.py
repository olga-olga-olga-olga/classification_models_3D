import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from datagenerator import datagen
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve

def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())  # avoid log(0)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * tf.pow(1. - pt, gamma)
        return -K.mean(focal_weight * tf.math.log(pt))
    return loss

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) \
               - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return focal_loss

def normalize_intensity(volume):
    """
    Apply intensity normalization - MUST match training data preprocessing
    Adjust this function based on your training preprocessing!
    """
    # Per-channel z-score normalization (common approach)
    for channel in range(volume.shape[-1]):
        channel_data = volume[..., channel]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            volume[..., channel] = (channel_data - mean) / std
    
    return volume

def convert_nifti_to_npy(nifti_file_tuples, output_dir, target_shape=(240, 240, 160)):
    """
    Convert NIfTI files to .npy format that datagen expects
    Args:
        nifti_file_tuples: List of tuples [(t1_path, t2_path, flair_path), ...]
        output_dir: Directory to save .npy files
        target_shape: Target volume dimensions
    Returns:
        List of .npy file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_paths = []
    
    print(f"Converting {len(nifti_file_tuples)} NIfTI file sets to .npy format...")
    
    for i, (t1_path, t2_path, flair_path) in enumerate(nifti_file_tuples):
        try:
            # Extract patient ID from filename (for Imago dataset structure)
            # Patient ID should match the folder name in the BIDS structure
            patient_id = os.path.basename(t1_path).split('_')[0]  # Gets 'sub-XXX' part
            # Alternative: extract from full path
            # patient_id = t1_path.split('/')[-4]  # Gets patient folder name from path
            
            print(f"Processing {i+1}/{len(nifti_file_tuples)}: {patient_id}")
            
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
            
            # Clean up memory
            del t1_data, t2_data, flair_data, volume
            
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    print(f"Successfully converted {len(npy_paths)} files to .npy format")
    return npy_paths

def find_nifti_files(nifti_data_dir, patient_ids):
    """
    Find corresponding T1, T2, and FLAIR NIfTI files for each patient
    Using Imago dataset structure: {patient_id}/ses-01/anat/*{modality}.nii.gz
    """
    from glob import glob
    nifti_file_tuples = []
    
    for patient_id in patient_ids:
        # Use the same patterns as Dataset2Handler
        t1_files = glob(f"{nifti_data_dir}/{patient_id}/ses-01/anat/*T1.nii.gz")
        t2_files = glob(f"{nifti_data_dir}/{patient_id}/ses-01/anat/*T2.nii.gz")
        flair_files = glob(f"{nifti_data_dir}/{patient_id}/ses-01/anat/*FLAIR.nii.gz")
        
        if len(t1_files) >= 1 and len(t2_files) >= 1 and len(flair_files) >= 1:
            # Take the first file if multiple found
            nifti_file_tuples.append((t1_files[0], t2_files[0], flair_files[0]))
            print(f"Found files for {patient_id}")
        else:
            print(f"Warning: Could not find complete set of files for {patient_id}")
            print(f"  T1: {len(t1_files)} files, T2: {len(t2_files)} files, FLAIR: {len(flair_files)} files")
            print(f"  Looking in: {nifti_data_dir}/{patient_id}/ses-01/anat/")
    
    return nifti_file_tuples

def main():
    # UPDATE THESE PATHS
    model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models/resnet34-0.0936-22.keras'
    nifti_data_dir = '/data/share/IMAGO/400_cases/IMAGO_501/bids/IMAGO_first_release/derivatives - registered/'
    excel_path = '/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xlsx'
    converted_npy_dir = '/home/radv/ofilipowicz/tmp/nifti_temp/'  # Temporary directory for converted files
    
    # Load data
    df = pd.read_excel(excel_path)
    
    # For Imago dataset, use 'Pseudo' column for patient IDs
    patient_ids = df['Pseudo'].astype(str).tolist()  # Convert to string to match folder names
    
    print("Converting NIfTI files to .npy format...")
    
    # Find NIfTI files for each patient
    nifti_file_tuples = find_nifti_files(nifti_data_dir, patient_ids)
    
    if len(nifti_file_tuples) == 0:
        print("ERROR: No matching NIfTI files found!")
        print("Please check:")
        print(f"1. NIfTI data directory: {nifti_data_dir}")
        print("2. File structure should be: {patient_id}/ses-01/anat/*{modality}.nii.gz")
        print("3. Patient IDs in Excel file match folder names")
        return
    
    # Convert NIfTI to .npy
    existing_files = convert_nifti_to_npy(nifti_file_tuples, converted_npy_dir)
    
    # Match labels to converted files
    converted_patient_ids = [os.path.basename(f).replace('.npy', '') for f in existing_files]
    existing_labels = []
    for converted_id in converted_patient_ids:
        # Find matching label in dataframe using 'Pseudo' column
        matching_rows = df[df['Pseudo'].astype(str) == converted_id]
        if len(matching_rows) > 0:
            # Extract IDH status based on tumor type (adjust based on your label extraction logic)
            tumor_type = str(matching_rows['Tumor_type'].iloc[0]).lower()
            if 'astrocytoma' in tumor_type:
                existing_labels.append(0)
            elif 'oligodendroglioma' in tumor_type:
                existing_labels.append(1)
            elif 'gbm' in tumor_type:
                existing_labels.append(2)
            else:
                print(f"Warning: Unknown tumor type '{tumor_type}' for {converted_id}")
                existing_labels.append(-1)  # Unknown label
        else:
            print(f"Warning: No label found for {converted_id}")
            existing_labels.append(-1)  # Missing label
    
    # Filter out cases with unknown/missing labels if needed
    valid_indices = [i for i, label in enumerate(existing_labels) if label != -1]
    existing_files = [existing_files[i] for i in valid_indices]
    existing_labels = [existing_labels[i] for i in valid_indices]
    
    print(f"Testing {len(existing_files)} files")
    
    if len(existing_files) == 0:
        print("ERROR: No files found for testing!")
        return
    
    # Load model and test
    model = load_model(model_path, custom_objects={'focal_loss': binary_focal_loss()})
    test_gen = datagen(existing_files, existing_labels, batch_size=1, shuffle=False, augment=False)
    
    predictions = model.predict(test_gen, steps=len(existing_files))
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    # Results
    accuracy = np.mean(pred_binary == existing_labels)
    auc = roc_auc_score(existing_labels, predictions.flatten())
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(existing_labels, predictions.flatten())
    tnr = 1 - fpr  # True Negative Rate (Specificity)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(existing_labels, predictions.flatten())
    
    # Confusion Matrix
    cm = confusion_matrix(existing_labels, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    print(f"\nResults on {len(existing_labels)} samples:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Sensitivity (TPR): {sensitivity:.3f}")
    print(f"Specificity (TNR): {specificity:.3f}")
    print(f"PPV (Precision): {ppv:.3f}")
    print(f"NPV: {npv:.3f}")
    print(f"FPR: {fp/(fp+tn):.3f}")
    print(f"FNR: {fn/(fn+tp):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(existing_labels, pred_binary, target_names=['IDH-wildtype', 'IDH-mutant']))
    
    # Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    ax1.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'r--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, 'g-', label=f'PR Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion Matrix
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['IDH-wt', 'IDH-mut'])
    ax3.set_yticklabels(['IDH-wt', 'IDH-mut'])
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    ax3.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    
    # Prediction Distribution
    ax4.hist(predictions[np.array(existing_labels) == 0], bins=20, alpha=0.7, 
             label='IDH-wildtype', color='blue')
    ax4.hist(predictions[np.array(existing_labels) == 1], bins=20, alpha=0.7, 
             label='IDH-mutant', color='red')
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Prediction Distribution')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plots
    output_plot_path = '/home/radv/ofilipowicz/my-scratch/test_results_plots.png'
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved to: {output_plot_path}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'file_path': existing_files,
        'patient_id': [os.path.basename(f).replace('.npy', '') for f in existing_files],
        'true_label': existing_labels,
        'predicted_probability': predictions.flatten(),
        'predicted_class': pred_binary
    })
    
    results_csv_path = '/home/radv/ofilipowicz/my-scratch/detailed_test_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"Detailed results saved to: {results_csv_path}")
    
    # Clean up temporary .npy files
    if os.path.exists(converted_npy_dir):
        import shutil
        print(f"\nCleaning up temporary files in {converted_npy_dir}")
        shutil.rmtree(converted_npy_dir)

if __name__ == "__main__":
    main()