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
from datetime import datetime
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

        # Debug: Print distribution of binary predictions
        print(f"IDH predictions distribution: Wildtype={np.sum(idh_binary==0)}, Mutant={np.sum(idh_binary==1)}")
        print(f"1p/19q predictions distribution: Not codeleted={np.sum(codeletion_binary==0)}, Codeleted={np.sum(codeletion_binary==1)}")
        
        # Detailed analysis of combinations
        idh_wt_1p19q_intact = np.sum((idh_binary==0) & (codeletion_binary==0))
        idh_wt_1p19q_codeleted = np.sum((idh_binary==0) & (codeletion_binary==1))
        idh_mut_1p19q_intact = np.sum((idh_binary==1) & (codeletion_binary==0))
        idh_mut_1p19q_codeleted = np.sum((idh_binary==1) & (codeletion_binary==1))
        
        print(f"\nDetailed combination analysis:")
        print(f"IDH-wildtype + 1p/19q-intact: {idh_wt_1p19q_intact} -> Glioblastoma")
        print(f"IDH-wildtype + 1p/19q-codeleted: {idh_wt_1p19q_codeleted} -> Glioblastoma (unusual but still GBM)")
        print(f"IDH-mutant + 1p/19q-intact: {idh_mut_1p19q_intact} -> Astrocytoma")
        print(f"IDH-mutant + 1p/19q-codeleted: {idh_mut_1p19q_codeleted} -> Oligodendroglioma")
        
        # Combine predictions according to WHO classification
        predictions = []
        probabilities = []
        classification_counts = {"glioblastoma": 0, "oligodendroglioma": 0, "astrocytoma": 0}

        for i in range(len(idh_binary)):
            if idh_binary[i] == 0:  # IDH wildtype
                pred_class = 0  # Glioblastoma
                classification_counts["glioblastoma"] += 1
                # Probability is confidence in IDH wildtype
                prob = [1 - idh_pred[i][0], 0, 0]
            elif idh_binary[i] == 1 and codeletion_binary[i] == 1:  # IDH mutant + codeleted
                pred_class = 1  # Oligodendroglioma
                classification_counts["oligodendroglioma"] += 1
                # Probability combines both model confidences
                prob = [0, idh_pred[i][0] * codeletion_pred[i][0], 0]
            else:  # IDH mutant + intact (or uncertain codeletion)
                pred_class = 2  # Astrocytoma
                classification_counts["astrocytoma"] += 1
                # Probability is IDH mutant confidence * (1 - codeletion confidence)
                prob = [0, 0, idh_pred[i][0] * (1 - codeletion_pred[i][0])]

            predictions.append(pred_class)
            probabilities.append(prob)

        # Print final classification distribution
        print(f"Final predictions distribution:")
        print(f"  Glioblastoma: {classification_counts['glioblastoma']}")
        print(f"  Oligodendroglioma: {classification_counts['oligodendroglioma']}")
        print(f"  Astrocytoma: {classification_counts['astrocytoma']}")

        return np.array(predictions), np.array(probabilities)

def get_true_three_class_labels(df):
    """
    Generate true three-class labels directly from the Tumor_type column.
    
    Args:
        df: Clinical data DataFrame with Tumor_type column
        
    Returns:
        Array of true labels (0=Glioblastoma, 1=Oligodendroglioma, 2=Astrocytoma)
    """
    true_labels = []

    for _, row in df.iterrows():
        tumor_type = row.get('Tumor_type', None)
        
        if pd.isna(tumor_type) or tumor_type == '' or tumor_type is None:
            true_labels.append(None)  # Exclude if no tumor type
            continue
            
        # Convert to string and normalize
        tumor_type_str = str(tumor_type).upper().strip()
        
        # Map tumor types to classes
        if any(keyword in tumor_type_str for keyword in ['GBM', 'GLIOBLASTOMA']):
            true_labels.append(0)  # Glioblastoma
        elif any(keyword in tumor_type_str for keyword in ['OLIGODENDROGLIOMA', 'OLIGO']):
            true_labels.append(1)  # Oligodendroglioma
        elif any(keyword in tumor_type_str for keyword in ['ASTROCYTOMA', 'ASTRO']):
            true_labels.append(2)  # Astrocytoma
        else:
            # Print unknown tumor types for debugging
            print(f"Unknown tumor type: '{tumor_type}' - excluding from analysis")
            true_labels.append(None)  # Exclude unknown types

    return true_labels

def plot_multiclass_roc(y_true, y_prob, class_names, save_path):
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_prob: Prediction probabilities for each class
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(y_true))
    n_classes = len(unique_classes)
    
    # Only use class names for classes that are actually present
    present_class_names = [class_names[i] for i in unique_classes]
    
    # Binarize the output for only the classes present in the data
    y_true_bin = label_binarize(y_true, classes=unique_classes)
    
    # If only 2 classes, label_binarize returns 1D array, need to reshape
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    # Extract probabilities for only the present classes
    y_prob_subset = y_prob[:, unique_classes]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob_subset[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob_subset.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{present_class_names[i]} (AUC = {roc_auc[i]:.3f})')

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Get unique classes present in the data
    present_labels = sorted(set(y_true) | set(y_pred))
    present_class_names = [class_names[i] for i in present_labels]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_class_names, 
                yticklabels=present_class_names)
    plt.title('Confusion Matrix - Three-Class Glioma Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_prediction_distribution(y_true, y_prob, class_names, save_path):
    """
    Plot prediction probability distributions for each class.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for class_idx in range(3):
        ax = axes[class_idx]
        
        # Get probabilities for this class
        class_probs = y_prob[:, class_idx]
        
        # Plot histogram for each true class
        colors = ['red', 'blue', 'green']
        for true_class in range(3):
            if np.sum(np.array(y_true) == true_class) > 0:
                mask = np.array(y_true) == true_class
                ax.hist(class_probs[mask], bins=20, alpha=0.7, 
                       label=f'True {class_names[true_class]}', 
                       color=colors[true_class])
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Distribution for {class_names[class_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_results_directory():
    """
    Create a timestamped results directory.
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'/home/radv/ofilipowicz/my-scratch/test_results/three_class_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def test_three_class_glioma_classification():
    """
    Test the three-class glioma classification using the same approach as binary testing.
    """
    # Create results directory
    results_dir = create_results_directory()
    print(f"Results will be saved to: {results_dir}")
    
    # PATHS - Update these to your actual model paths
    idh_model_path = '/home/radv/ofilipowicz/my-scratch/olga/densenet121_2-0.0546-12.keras'
    codeletion_model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_1cat_1q19p/run_20250714_104621/densenet169--1.1825-10.keras'
    data_path = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl/'
    excel_path = '/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xlsx'

    # Load data
    print("Loading clinical data...")
    df_imago = pd.read_excel(excel_path, engine='openpyxl')
    print(f"Found {len(df_imago)} IMAGO samples")

    # Create file paths for IMAGO samples only
    test_files = [f'{data_path}/{pid}.npy' for pid in df_imago['Pseudo']]

    # Get true three-class labels for IMAGO samples only
    print("Analyzing tumor types in dataset...")
    tumor_types = df_imago['Tumor_type'].dropna().unique()
    print(f"Available tumor types: {tumor_types}")
    
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

    # Initialize classifier
    classifier = ThreeClassGliomaClassifier(idh_model_path, codeletion_model_path)

    # Create data generator
    test_gen = datagen(existing_files, existing_labels, batch_size=1, shuffle=False, augment=False)

    # Make predictions
    print("Making three-class predictions...")
    predictions, probabilities = classifier.predict_three_class(test_gen, len(existing_files))

    # Calculate metrics
    accuracy = accuracy_score(existing_labels, predictions)
    unique_true_labels = sorted(np.unique(existing_labels))
    unique_predicted_labels = sorted(np.unique(predictions))
    
    print(f"\nThree-Class Glioma Classification Results:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classes present in true labels: {[class_names[i] for i in unique_true_labels]}")
    print(f"Classes present in predictions: {[class_names[i] for i in unique_predicted_labels]}")
    
    # Calculate per-class accuracy scores
    print(f"\nPer-Class Accuracy Scores:")
    per_class_accuracies = {}
    
    for class_idx in unique_true_labels:
        class_name = class_names[class_idx]
        class_mask = np.array(existing_labels) == class_idx
        
        if np.sum(class_mask) > 0:
            class_predictions = np.array(predictions)[class_mask]
            class_true_labels = np.array(existing_labels)[class_mask]
            class_accuracy = accuracy_score(class_true_labels, class_predictions)
            per_class_accuracies[class_idx] = class_accuracy
            
            correct_predictions = np.sum(class_predictions == class_idx)
            total_samples = len(class_true_labels)
            
            print(f"  {class_name}: {class_accuracy:.3f} ({correct_predictions}/{total_samples} correct)")
        else:
            per_class_accuracies[class_idx] = 0.0
            print(f"  {class_name}: No samples in test set")
    
    # Calculate balanced accuracy
    present_class_accuracies = [acc for class_idx, acc in per_class_accuracies.items() 
                               if class_idx in unique_true_labels and np.sum(np.array(existing_labels) == class_idx) > 0]
    balanced_accuracy = np.mean(present_class_accuracies) if present_class_accuracies else 0.0
    print(f"\nBalanced Accuracy (average per-class): {balanced_accuracy:.3f}")
    
    # Get present labels for classification report
    present_labels = sorted(set(existing_labels) | set(predictions))
    present_class_names = [class_names[i] for i in present_labels]
    
    print(f"\nClassification Report:")
    class_report = classification_report(existing_labels, predictions, 
                                       labels=present_labels,
                                       target_names=present_class_names)
    print(class_report)

    # Generate and save plots
    print("\nGenerating and saving plots...")
    
    # ROC curves
    roc_save_path = os.path.join(results_dir, 'roc_curves.png')
    roc_auc_scores = plot_multiclass_roc(existing_labels, probabilities, 
                                        classifier.class_labels, roc_save_path)

    # Confusion matrix
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    cm = plot_confusion_matrix(existing_labels, predictions, 
                              classifier.class_labels, cm_save_path)

    # Prediction distribution
    pred_dist_save_path = os.path.join(results_dir, 'prediction_distribution.png')
    plot_prediction_distribution(existing_labels, probabilities, 
                                classifier.class_labels, pred_dist_save_path)

    print(f"\nAUC Scores:")
    for i, class_name in enumerate(classifier.class_labels):
        if i in roc_auc_scores:
            print(f"{class_name}: {roc_auc_scores[i]:.3f}")
        else:
            print(f"{class_name}: Not present in data")
    print(f"Micro-average: {roc_auc_scores['micro']:.3f}")

    # Save comprehensive results to text file
    metrics_path = os.path.join(results_dir, 'classification_results.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("THREE-CLASS GLIOMA CLASSIFICATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Test Dataset: IMAGO ({len(existing_labels)} samples)\n")
        f.write(f"IDH Model: {os.path.basename(idh_model_path)}\n")
        f.write(f"1p/19q Model: {os.path.basename(codeletion_model_path)}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.3f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy:.3f}\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        for i, class_name in enumerate(classifier.class_labels):
            count = np.sum(np.array(existing_labels) == i)
            percentage = count / len(existing_labels) * 100
            f.write(f"{class_name}: {count} samples ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("PER-CLASS ACCURACY:\n")
        f.write("-" * 20 + "\n")
        for class_idx in unique_true_labels:
            class_name = class_names[class_idx]
            class_mask = np.array(existing_labels) == class_idx
            if np.sum(class_mask) > 0:
                class_predictions = np.array(predictions)[class_mask]
                correct_predictions = np.sum(class_predictions == class_idx)
                total_samples = len(class_predictions)
                class_accuracy = per_class_accuracies[class_idx]
                f.write(f"{class_name}: {class_accuracy:.3f} ({correct_predictions}/{total_samples} correct)\n")
            else:
                f.write(f"{class_name}: No samples in test set\n")
        f.write("\n")

        f.write("AUC SCORES:\n")
        f.write("-" * 20 + "\n")
        for i, class_name in enumerate(classifier.class_labels):
            if i in roc_auc_scores:
                f.write(f"{class_name}: {roc_auc_scores[i]:.3f}\n")
            else:
                f.write(f"{class_name}: Not present in data\n")
        f.write(f"Micro-average: {roc_auc_scores['micro']:.3f}\n\n")

        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 20 + "\n")
        f.write(class_report)
        f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        f.write("Rows: True labels, Columns: Predicted labels\n")
        f.write("Classes: " + ", ".join([f"{i}={name}" for i, name in enumerate(present_class_names)]) + "\n")
        f.write(str(cm) + "\n\n")

        f.write("FILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write("- roc_curves.png: ROC curves for all classes\n")
        f.write("- confusion_matrix.png: Confusion matrix heatmap\n")
        f.write("- prediction_distribution.png: Prediction probability distributions\n")
        f.write("- classification_results.txt: This comprehensive report\n")

    print(f"\nAll results saved to: {results_dir}")
    print(f"Files generated:")
    print(f"  - ROC curves: {roc_save_path}")
    print(f"  - Confusion matrix: {cm_save_path}")
    print(f"  - Prediction distribution: {pred_dist_save_path}")
    print(f"  - Results report: {metrics_path}")

    return accuracy, predictions, probabilities, roc_auc_scores, results_dir

# Example usage
if __name__ == "__main__":
    accuracy, predictions, probabilities, auc_scores, results_dir = test_three_class_glioma_classification()