import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from datagenerator import datagen

class ThreeClassGliomaClassifier:
    def __init__(self, idh_model_path, codeletion_model_path):
        """
        Initialize the three-class classifier with paths to the two binary models.
        
        Args:
            idh_model_path: Path to the IDH mutation classification model
            codeletion_model_path: Path to the 1p/19q codeletion classification model
        """
        # Load models with custom objects
        print("Loading IDH model...")
        self.idh_model = load_model(idh_model_path, compile=False)
        print("Loading 1p/19q model...")
        self.codeletion_model = load_model(codeletion_model_path, compile=False)
        
        # Class mapping
        self.class_names = {0: 'Glioblastoma', 1: 'Oligodendroglioma', 2: 'Astrocytoma'}
        self.class_labels = ['Glioblastoma', 'Oligodendroglioma', 'Astrocytoma']
    
    def predict_three_class(self, data_generator, num_samples):
        """
        Predict three-class cancer type based on IDH and 1p/19q predictions.
        
        Args:
            data_generator: Data generator for the test samples
            num_samples: Number of samples to process
            
        Returns:
            predictions: Array of class predictions (0, 1, or 2)
            probabilities: Array of prediction probabilities for each class
        """
        print("Getting IDH predictions...")
        idh_pred = self.idh_model.predict(data_generator, steps=num_samples)
        
        # Reset generator for second model
        print("Getting 1p/19q predictions...")
        codeletion_pred = self.codeletion_model.predict(data_generator, steps=num_samples)
        
        # Convert to binary predictions (threshold at 0.5)
        idh_binary = (idh_pred > 0.5).astype(int).flatten()
        codeletion_binary = (codeletion_pred > 0.5).astype(int).flatten()
        
        # Combine predictions according to WHO classification
        predictions = []
        probabilities = []
        
        for i in range(len(idh_binary)):
            if idh_binary[i] == 0:  # IDH wildtype
                pred_class = 0  # Glioblastoma
                # Probability is confidence in IDH wildtype
                prob = [1 - idh_pred[i][0], 0, 0]
            elif idh_binary[i] == 1 and codeletion_binary[i] == 1:  # IDH mutant + codeleted
                pred_class = 1  # Oligodendroglioma
                # Probability combines both model confidences
                prob = [0, idh_pred[i][0] * codeletion_pred[i][0], 0]
            else:  # IDH mutant + intact (or uncertain codeletion)
                pred_class = 2  # Astrocytoma
                # Probability is IDH mutant confidence * (1 - codeletion confidence)
                prob = [0, 0, idh_pred[i][0] * (1 - codeletion_pred[i][0])]
            
            predictions.append(pred_class)
            probabilities.append(prob)
        
        return np.array(predictions), np.array(probabilities)

def get_true_three_class_labels(df):
    """
    Generate true three-class labels from the clinical data DataFrame.
    
    Args:
        df: Clinical data DataFrame with IDH, 1p/19q, and Tumor_type columns
        
    Returns:
        Array of true labels (0=Glioblastoma, 1=Oligodendroglioma, 2=Astrocytoma)
    """
    true_labels = []
    
    for _, row in df.iterrows():
        idh_status = row.get('IDH', None)  # Assuming IDH column exists
        codeletion_status = row.get('1p/19q', None)  # 1p/19q column
        tumor_type = row.get('Tumor_type', None)  # Tumor type column
        
        # Handle GBM cases first - if it's GBM, it's always not codeleted
        if tumor_type and 'GBM' in str(tumor_type).upper():
            true_labels.append(0)  # Glioblastoma
            continue
        
        # Handle codeletion status
        if pd.isna(codeletion_status) or codeletion_status == '' or codeletion_status is None:
            # Unknown codeletion status
            if tumor_type and 'GBM' in str(tumor_type).upper():
                true_labels.append(0)  # GBM = Glioblastoma
            else:
                true_labels.append(None)  # Exclude if unknown
            continue
        
        # Process based on codeletion status
        codeletion_str = str(codeletion_status).lower().strip()
        
        if 'codeleted' in codeletion_str and 'not' not in codeletion_str:
            # "codeleted" = Oligodendroglioma (assuming IDH mutant)
            true_labels.append(1)
        elif 'not codeleted' in codeletion_str or 'intact' in codeletion_str:
            # "not codeleted" could be Glioblastoma (IDH-wt) or Astrocytoma (IDH-mut)
            # We need IDH status to distinguish
            if pd.isna(idh_status):
                # If no IDH info, assume based on tumor type
                if tumor_type and 'GBM' in str(tumor_type).upper():
                    true_labels.append(0)  # Glioblastoma
                else:
                    true_labels.append(2)  # Default to Astrocytoma for IDH unknown
            elif idh_status == 0 or str(idh_status).lower() == 'wt IDH':
                true_labels.append(0)  # Glioblastoma
            elif idh_status == 1 or str(idh_status).lower() == 'IDH 1 mutation':
                true_labels.append(2)  # Astrocytoma
            else:
                true_labels.append(None)  # Exclude uncertain cases
        else:
            true_labels.append(None)  # Exclude uncertain cases
    
    return true_labels

def plot_multiclass_roc(y_true, y_prob, class_names):
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_prob: Prediction probabilities for each class
        class_names: List of class names
    """
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Three-Class Glioma Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc

def test_three_class_glioma_classification():
    """
    Test the three-class glioma classification using the same approach as binary testing.
    """
    # PATHS - Update these to your actual model paths
    idh_model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_1cat/run_YYYYMMDD_HHMMSS/densenet169-best.keras'
    codeletion_model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_1cat_1q19p/run_20250711_211954/densenet169-1.8408-23.keras'
    data_path = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl/'
    excel_path = '/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xlsx'
    
    # Load data
    print("Loading clinical data...")
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Filter for IMAGO dataset only
    print("Filtering for IMAGO dataset...")
    df_imago = df[df['Dataset'] == 'IMAGO'].copy()
    print(f"Found {len(df_imago)} IMAGO samples out of {len(df)} total samples")
    
    if len(df_imago) == 0:
        print("No IMAGO samples found! Check the Dataset column values.")
        print("Available datasets:", df['Dataset'].unique())
        return
    
    # Create file paths for IMAGO samples only
    test_files = [f'{data_path}/{pid}.npy' for pid in df_imago['Pseudo']]
    
    # Get true three-class labels for IMAGO samples only
    true_labels = get_true_three_class_labels(df_imago)
    
    # Filter existing files and valid labels
    existing_data = []
    for i, (file_path, label) in enumerate(zip(test_files, true_labels)):
        if os.path.exists(file_path) and label is not None:
            existing_data.append((file_path, label))
    
    if not existing_data:
        print("No valid test samples found!")
        return
    
    existing_files, existing_labels = zip(*existing_data)
    existing_files = list(existing_files)
    existing_labels = list(existing_labels)
    
    print(f"Testing {len(existing_files)} files")
    
    # Show label distribution with names
    label_counts = np.bincount(existing_labels)
    class_names = ['Glioblastoma', 'Oligodendroglioma', 'Astrocytoma']
    print("Label distribution:")
    for i, count in enumerate(label_counts):
        if count > 0:
            print(f"  {class_names[i]}: {count}")
    
    # Debug: Show some examples of label assignment for IMAGO samples
    print("\nFirst 5 IMAGO label assignments (for debugging):")
    debug_labels = get_true_three_class_labels(df_imago)
    for i in range(min(5, len(df_imago))):
        row = df_imago.iloc[i]
        print(f"  Patient {row.get('Pseudo', 'Unknown')}: IDH={row.get('IDH', 'Missing')}, "
              f"1p/19q='{row.get('1p/19q', 'Missing')}', Tumor_type='{row.get('Tumor_type', 'Missing')}' "
              f"-> Label={debug_labels[i]} ({class_names[debug_labels[i]] if debug_labels[i] is not None else 'Excluded'})")
    
    # Initialize classifier
    classifier = ThreeClassGliomaClassifier(idh_model_path, codeletion_model_path)
    
    # Create data generator (same approach as binary testing)
    test_gen = datagen(existing_files, existing_labels, batch_size=1, shuffle=False, augment=False)
    
    # Make predictions
    print("Making three-class predictions...")
    predictions, probabilities = classifier.predict_three_class(test_gen, len(existing_files))
    
    # Calculate metrics
    accuracy = accuracy_score(existing_labels, predictions)
    
    print(f"\nThree-Class Glioma Classification Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(existing_labels, predictions, 
                              target_names=classifier.class_labels))
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    roc_auc_scores = plot_multiclass_roc(existing_labels, probabilities, classifier.class_labels)
    
    print(f"\nAUC Scores:")
    for i, class_name in enumerate(classifier.class_labels):
        print(f"{class_name}: {roc_auc_scores[i]:.3f}")
    print(f"Micro-average: {roc_auc_scores['micro']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(existing_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.class_labels, 
                yticklabels=classifier.class_labels)
    plt.title('Confusion Matrix - Three-Class Glioma Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    
    # Save results
    os.makedirs('/home/radv/ofilipowicz/my-scratch/test_results/', exist_ok=True)
    
    # Save metrics to a text file
    metrics_path = '/home/radv/ofilipowicz/my-scratch/test_results/three_class_glioma_results_IMAGO.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Three-Class Glioma Classification Results on IMAGO Dataset ({len(existing_labels)} samples):\n")
        f.write(f"Accuracy: {accuracy:.3f}\n\n")
        
        f.write("AUC Scores:\n")
        for i, class_name in enumerate(classifier.class_labels):
            f.write(f"{class_name}: {roc_auc_scores[i]:.3f}\n")
        f.write(f"Micro-average: {roc_auc_scores['micro']:.3f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(existing_labels, predictions, 
                                    target_names=classifier.class_labels))
        
        f.write(f"\nLabel distribution:\n")
        for i, class_name in enumerate(classifier.class_labels):
            count = np.sum(np.array(existing_labels) == i)
            f.write(f"{class_name}: {count}\n")
    
    print(f"\nResults saved to: {metrics_path}")
    
    return accuracy, predictions, probabilities, roc_auc_scores

# Example usage
if __name__ == "__main__":
    accuracy, predictions, probabilities, auc_scores = test_three_class_glioma_classification()
    
    print(f"\nFinal three-class accuracy: {accuracy:.4f}")
    print(f"Average AUC: {np.mean([auc_scores[i] for i in range(3)]):.3f}")