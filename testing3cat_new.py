import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

# Import your custom utilities
from dataloader import create_batch_generators
from dataset import Datasets, Dataset3Handler

# Only include custom functions if you need to calculate loss/metrics during loading
# For predictions only, these aren't needed!

def get_predictions_and_labels(generator, model, max_samples=774):
    """Get predictions and true labels with sample limit"""
    print(f"Getting predictions from {max_samples} samples...")
    
    all_true_labels = []
    all_predictions = []
    batch_count = 0
    
    while len(all_true_labels) < max_samples:
        batch = next(generator)
        X_batch, y_batch = batch
        
        # Get predictions
        pred_batch = model.predict(X_batch, verbose=0)
        
        # Store true labels (convert from one-hot to class indices)
        true_batch = np.argmax(y_batch, axis=1)
        all_true_labels.extend(true_batch)
        all_predictions.extend(pred_batch)
        
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"Processed {batch_count} batches, {len(all_true_labels)} samples")
    
    # Trim to exact size
    all_true_labels = all_true_labels[:max_samples]
    all_predictions = all_predictions[:max_samples]
    
    print(f"Finished! {len(all_true_labels)} samples processed")
    
    y_true = np.array(all_true_labels)
    y_pred_proba = np.array(all_predictions)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_true, y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix with per-class accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add per-class accuracy on diagonal
    for i in range(len(class_names)):
        if cm[i, :].sum() > 0:
            accuracy = cm[i, i] / cm[i, :].sum()
            plt.text(i + 0.5, i - 0.3, f'{accuracy:.3f}', 
                    ha='center', va='center', fontweight='bold', color='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_all_metrics(y_true, y_pred, y_pred_proba, class_names):
    """Calculate all metrics from the same predictions"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Per-class accuracy (like confusion matrix diagonal)
    print("\nPer-Class Accuracy (Multi-class):")
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            print(f"  {class_name}: {class_acc:.4f}")
    
    # Binary accuracy for each class (like your custom functions)
    print("\nPer-Class Binary Accuracy (Class vs Rest):")
    for i, class_name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        binary_acc = accuracy_score(y_true_bin, y_pred_bin)
        print(f"  {class_name}: {binary_acc:.4f}")
    
    # Class distribution
    print(f"\nClass Distribution: {np.bincount(y_true)}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix numbers
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"Classes: {class_names}")
    print(cm)

if __name__ == '__main__':
    # --- CONFIGURATION ---
    model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_3cat/run_20250707_190555/resnet18-0.6492-13.keras'
    data_path = "/data/share/IMAGO/Rotterdam/"
    excel_path = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"
    output_dir = '/home/radv/ofilipowicz/my-scratch/test_results/'
    
    # Parameters
    batch_size = 1
    num_classes = 3
    class_names = ['Astrocytoma', 'Oligodendroglioma', 'GBM']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # --- LOAD DATA (ONCE) ---
    print("Setting up dataset...")
    datasets = Datasets(target_size=(240, 240, 189), target_spacing=(1.0, 1.0, 1.0))
    datasets.add_dataset(data_path, excel_path, Dataset3Handler)
    
    # Create generator (ONCE)
    _, gen_test, _ = create_batch_generators(
        datasets,
        batch_size_train=batch_size,
        batch_size_valid=batch_size,
        target_size=(240, 240, 189),
        num_channels=3
    )
    
    # --- LOAD MODEL (simplified for predictions only) ---
    print("Loading model...")
    # For predictions only, you can often load without custom objects
    try:
        model = load_model(model_path, compile=False)  # compile=False skips loss/metrics
        print("Model loaded without custom objects (compile=False)")
    except:
        # If that fails, include custom objects
        alpha_weights = [1.0, 3.0, 0.6]
        custom_objects = {
            'focal_loss': categorical_focal_loss(gamma=2.0, alpha=alpha_weights),
            'class_0_accuracy': class_0_accuracy,
            'class_1_accuracy': class_1_accuracy,
            'class_2_accuracy': class_2_accuracy
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded with custom objects")
    
    # --- GET PREDICTIONS FROM EXACTLY 774 SAMPLES ---
    y_true, y_pred, y_pred_proba = get_predictions_and_labels(gen_test, model, max_samples=774)
    
    # --- CALCULATE ALL METRICS FROM SAME DATA ---
    calculate_all_metrics(y_true, y_pred, y_pred_proba, class_names)
    
    # --- PLOT CONFUSION MATRIX ---
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    )
    
    # --- SAVE RESULTS ---
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_class': [class_names[i] for i in y_true],
        'predicted_class': [class_names[i] for i in y_pred],
        **{f'prob_{class_names[i]}': y_pred_proba[:, i] for i in range(num_classes)}
    })
    
    results_df.to_csv(os.path.join(output_dir, f'predictions_{model_name}.csv'), index=False)
    print(f"\nResults saved to: {output_dir}")
    print(f"Files: confusion_matrix_{model_name}.png, predictions_{model_name}.csv")