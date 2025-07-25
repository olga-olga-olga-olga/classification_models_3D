import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.models import Model
from keras import backend as K

# Import your custom utilities and metrics
from dataloader import create_batch_generators, get_preprocess_input_dummy
from dataset import Datasets, Dataset1Handler, Dataset2Handler, Dataset3Handler

# Custom loss and metrics (must match those used in training)
def categorical_focal_loss(gamma=2., alpha=None):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true * alpha, axis=-1)
            ce = alpha_t * ce
        focal_term = tf.pow(1 - p_t, gamma)
        loss = focal_term * ce
        return tf.reduce_mean(loss)
    return focal_loss

def class_0_accuracy(y_true, y_pred):
    class_id = 0
    return tf.keras.metrics.binary_accuracy(y_true[:, class_id], y_pred[:, class_id])

def class_1_accuracy(y_true, y_pred):
    class_id = 1
    return tf.keras.metrics.binary_accuracy(y_true[:, class_id], y_pred[:, class_id])

def class_2_accuracy(y_true, y_pred):
    class_id = 2
    return tf.keras.metrics.binary_accuracy(y_true[:, class_id], y_pred[:, class_id])

def extract_true_labels_and_predictions(generator, model, steps, num_classes=3):
    """Extract true labels and predictions from generator"""
    print("Extracting predictions and true labels...")
    
    all_true_labels = []
    all_predictions = []
    
    for i in range(steps):
        try:
            batch = next(generator)
            X_batch, y_batch = batch
            
            # Get predictions for this batch
            pred_batch = model.predict(X_batch, verbose=0)
            
            # Store true labels (convert from one-hot to class indices)
            true_batch = np.argmax(y_batch, axis=1)
            all_true_labels.extend(true_batch)
            
            # Store predictions (probabilities)
            all_predictions.extend(pred_batch)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{steps} batches")
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            break
    
    true_labels = np.array(all_true_labels)
    predictions = np.array(all_predictions)
    
    # Ensure predictions are properly normalized (probabilities sum to 1)
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    
    print(f"Extracted {len(true_labels)} samples")
    print(f"True labels range: {np.min(true_labels)} to {np.max(true_labels)}")
    print(f"Predictions shape: {predictions.shape}")
    
    return true_labels, predictions

def debug_predictions(y_true, y_pred, y_pred_proba, class_names):
    """Debug function to check prediction consistency"""
    print("Debugging predictions...")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_pred_proba shape: {y_pred_proba.shape}")
    
    print(f"\ny_true range: {np.min(y_true)} to {np.max(y_true)}")
    print(f"y_pred range: {np.min(y_pred)} to {np.max(y_pred)}")
    
    print(f"\nTrue label distribution: {np.bincount(y_true)}")
    print(f"Predicted label distribution: {np.bincount(y_pred)}")
    
    # Check if argmax of probabilities matches y_pred
    y_pred_from_proba = np.argmax(y_pred_proba, axis=1)
    matches = np.sum(y_pred == y_pred_from_proba)
    print(f"\nConsistency check - y_pred matches argmax(y_pred_proba): {matches}/{len(y_pred)} ({matches/len(y_pred)*100:.1f}%)")
    
    # Show some examples
    print(f"\nFirst 10 samples:")
    for i in range(min(10, len(y_true))):
        print(f"Sample {i}: True={y_true[i]} ({class_names[y_true[i]]}), "
              f"Pred={y_pred[i]} ({class_names[y_pred[i]]}), "
              f"Proba={y_pred_proba[i]}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix with correct metrics"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate overall accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nOverall Accuracy: {overall_accuracy:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add per-class metrics
    for i in range(len(class_names)):
        # Recall (Sensitivity): True Positives / (True Positives + False Negatives)
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        
        # Precision: True Positives / (True Positives + False Positives)  
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        
        # Add text annotations
        plt.text(i + 0.5, i - 0.3, f'Recall: {recall:.2f}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
        plt.text(i + 0.5, i + 0.3, f'Prec: {precision:.2f}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed confusion matrix analysis
    print("\nConfusion Matrix Analysis:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Total samples: {np.sum(cm)}")
    print("\nPer-class breakdown:")
    
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        false_negatives = cm[i, :].sum() - cm[i, i]  # Row sum - diagonal
        false_positives = cm[:, i].sum() - cm[i, i]  # Column sum - diagonal
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  True Positives: {true_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  False Positives: {false_positives}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
    
    return cm

def plot_roc_curves(y_true, y_pred_proba, num_classes, class_names, save_path=None):
    """Plot ROC curves for each class (one-vs-rest)"""
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(y_true, y_pred_proba, num_classes, class_names, save_path=None):
    """Plot Precision-Recall curves for each class"""
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(num_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision = precision_score(y_true_bin[:, i], y_pred_proba[:, i] > 0.5)
        
        plt.plot(recall, precision, color=color, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_distributions(y_true, y_pred_proba, num_classes, class_names, save_path=None):
    """Plot prediction probability distributions for each class"""
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    
    for i in range(num_classes):
        ax = axes[i] if num_classes > 1 else axes
        
        # Plot distributions for each true class
        for true_class in range(num_classes):
            mask = y_true == true_class
            if np.sum(mask) > 0:
                ax.hist(y_pred_proba[mask, i], bins=20, alpha=0.7, 
                       label=f'True {class_names[true_class]}')
        
        ax.set_xlabel(f'Predicted Probability for {class_names[i]}')
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Distribution - {class_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance_summary(y_true, y_pred, y_pred_proba, class_names, save_path=None):
    """Plot summary of per-class performance metrics"""
    metrics_data = []
    
    for i, class_name in enumerate(class_names):
        # Calculate metrics for each class
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        
        # AUC for binary classification (class vs rest)
        if len(np.unique(y_true_bin)) > 1:
            auc = roc_auc_score(y_true_bin, y_pred_proba[:, i])
        else:
            auc = 0.0
        
        metrics_data.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Plot bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_metrics['Class'], df_metrics[metric], alpha=0.7)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Class')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_metrics

if __name__ == '__main__':
    # --- CONFIGURE THESE PATHS ---
    model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_3cat/run_20250714_182554/densenet169-0.4423-46.keras'  
    data_path = "/data/share/IMAGO/Rotterdam/"          
    excel_path = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"   
    output_dir = '/home/radv/ofilipowicz/my-scratch/test_results/'  # Directory to save plots

    # Extract model name (without extension) for file naming
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- PREPARE DATASET AND GENERATOR ---
    datasets = Datasets(target_size=(240, 240, 189), target_spacing=(1.0, 1.0, 1.0))
    # Use the appropriate handler for your external data
    datasets.add_dataset(data_path, excel_path, Dataset3Handler)  # Change handler if needed

    num_classes = 3
    shape_size = (240, 240, 189, 3)
    batch_size = 1  
    class_names = ['Astrocytoma', 'Oligodendroglioma', 'GBM']  # Update with your class names
    steps = 750
    

    # Create the generator once
    _, gen_test, _ = create_batch_generators(
        datasets,
        batch_size_train=batch_size,
        batch_size_valid=batch_size,
        target_size=(240, 240, 189),       
        num_channels=3
    )

    # --- LOAD MODEL WITH CUSTOM OBJECTS ---
    alpha_weights = [1.0, 3.0, 0.6]  # Use the same as in training
    custom_objects = {
        'focal_loss': categorical_focal_loss(gamma=2.0, alpha=alpha_weights),
        'class_0_accuracy': class_0_accuracy,
        'class_1_accuracy': class_1_accuracy,
        'class_2_accuracy': class_2_accuracy
    }
    model = load_model(model_path, custom_objects=custom_objects)

    # --- EVALUATE ---
    print("Evaluating model...")
    results = model.evaluate(gen_test, steps=steps, verbose=1)
    print("Test results (loss, acc, class_0_acc, class_1_acc, class_2_acc):", results)

    # --- GET PREDICTIONS AND TRUE LABELS ---
    # Reset generator
    _, gen_test_pred, _ = create_batch_generators(
        datasets,
        batch_size_train=batch_size,
        batch_size_valid=batch_size,
        target_size=(240, 240, 189),       
        num_channels=3
    )
    y_true, y_pred_proba = extract_true_labels_and_predictions(gen_test_pred, model, steps, num_classes)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"\nExtracted {len(y_true)} samples")
    print(f"Class distribution: {np.bincount(y_true)}")

    # --- DEBUG PREDICTIONS ---
    debug_predictions(y_true, y_pred, y_pred_proba, class_names)

    # --- GENERATE COMPREHENSIVE PLOTS ---
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(
        y_true, y_pred, class_names, 
        save_path=os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    )
    
    # 2. ROC Curves
    print("Generating ROC curves...")
    plot_roc_curves(
        y_true, y_pred_proba, num_classes, class_names,
        save_path=os.path.join(output_dir, f'roc_curves_{model_name}.png')
    )
    
    # 3. Precision-Recall Curves
    print("Generating Precision-Recall curves...")
    plot_precision_recall_curves(
        y_true, y_pred_proba, num_classes, class_names,
        save_path=os.path.join(output_dir, f'pr_curves_{model_name}.png')
    )
    
    # 4. Prediction Distributions
    print("Generating prediction distributions...")
    plot_prediction_distributions(
        y_true, y_pred_proba, num_classes, class_names,
        save_path=os.path.join(output_dir, f'prediction_distributions_{model_name}.png')
    )
    
    # 5. Class Performance Summary
    print("Generating performance summary...")
    df_metrics = plot_class_performance_summary(
        y_true, y_pred, y_pred_proba, class_names,
        save_path=os.path.join(output_dir, f'class_performance_{model_name}.png')
    )
    
    # --- PRINT DETAILED RESULTS ---
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION RESULTS")
    print("="*50)
    
    # Overall accuracy
    overall_accuracy = np.mean(y_pred == y_true)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(df_metrics.to_string(index=False))
    
    # Confusion matrix numbers
    print(f"\nConfusion Matrix:")
    print(f"Classes: {class_names}")
    print(cm)
    
    # Save detailed results to CSV
    df_metrics.to_csv(os.path.join(output_dir, f'detailed_metrics_{model_name}.csv'), index=False)
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        **{f'prob_{class_names[i]}': y_pred_proba[:, i] for i in range(num_classes)}
    })
    results_df.to_csv(os.path.join(output_dir, f'predictions_{model_name}.csv'), index=False)
    
    print(f"\nAll results saved to: {output_dir}")
    print("Generated files:")
    print(f"- confusion_matrix_{model_name}.png")
    print(f"- roc_curves_{model_name}.png")
    print(f"- pr_curves_{model_name}.png")
    print(f"- prediction_distributions_{model_name}.png")
    print(f"- class_performance_{model_name}.png")
    print(f"- detailed_metrics_{model_name}.csv")
    print(f"- predictions_{model_name}.csv")