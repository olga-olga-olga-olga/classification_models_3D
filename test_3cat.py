# coding: utf-8
"""
Simple test script for evaluating a trained 3D classification model
on an external dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import create_batch_generators, get_preprocess_input_dummy
from dataset import Datasets, Dataset1Handler, Dataset2Handler, Dataset3Handler
import tensorflow as tf
from keras import backend as K


def categorical_focal_loss(gamma=2., alpha=None):
    """
    Focal loss function - must be defined for loading the trained model
    """
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


def load_trained_model(model_path):
    """
    Load a trained model
    """
    try:
        # First try loading without custom objects
        model = load_model(model_path)
        print("Model loaded without custom objects")
        return model
    except Exception as e:
        print(f"Failed to load without custom objects: {e}")
        print("Trying with custom objects...")
        
        # Define custom objects for loading if needed
        custom_objects = {
            'focal_loss': categorical_focal_loss(gamma=2.1, alpha=[1.2, 1.2, 1.0]),
            'class_0_accuracy': class_0_accuracy,
            'class_1_accuracy': class_1_accuracy,
            'class_2_accuracy': class_2_accuracy
        }
        
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded with custom objects")
        return model


def prepare_test_dataset(data_path, excel_path, dataset_handler_class):
    """
    Prepare test dataset using the same preprocessing as training
    """
    # Shape parameters (should match training)
    target_size = (240, 240, 189)
    target_spacing = (1.0, 1.0, 1.0)
    num_channels = 3
    
    # Initialize datasets
    datasets = Datasets(
        target_size=target_size,
        target_spacing=target_spacing,
        num_channels=num_channels
    )
    
    # Add test dataset
    datasets.add_dataset(data_path, excel_path, dataset_handler_class)
    
    # Create test generator
    gen_test, _, _ = create_batch_generators(
        datasets,
        batch_size_train=1,
        batch_size_valid=1,
        target_size=target_size,
        target_spacing=target_spacing,
        num_channels=num_channels,
        test_only=True  # You might need to modify create_batch_generators to support this
    )
    
    return gen_test, datasets


def evaluate_model(model, test_generator, num_test_samples):
    """
    Evaluate model on test dataset
    """
    print("Evaluating model on test dataset...")
    
    # Predict on test data
    predictions = []
    true_labels = []
    
    steps = min(num_test_samples, 100)  # Limit for demonstration
    
    for i in range(steps):
        try:
            batch_x, batch_y = next(test_generator)
            pred = model.predict(batch_x, verbose=0)
            
            predictions.append(pred[0])
            true_labels.append(batch_y[0])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{steps} samples")
                
        except StopIteration:
            break
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Convert to class indices
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    return predictions, true_labels, pred_classes, true_classes


def plot_confusion_matrix(y_true, y_pred, class_names=['Class 0', 'Class 1', 'Class 2']):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(predictions, true_classes):
    """
    Plot prediction confidence distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for class_id in range(3):
        class_mask = true_classes == class_id
        if np.any(class_mask):
            confidences = predictions[class_mask, class_id]
            axes[class_id].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            axes[class_id].set_title(f'Class {class_id} Confidence Distribution')
            axes[class_id].set_xlabel('Confidence')
            axes[class_id].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main testing function
    """
    
    model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_3cat/run_20250710_105454/densenet169-1.2295-35.keras'  
    test_data_path = "/data/share/IMAGO/Rotterdam/"          
    test_excel_path = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"   
    

    
    # Choose appropriate dataset handler based on your test data format
    # Use Dataset1Handler, Dataset2Handler, or Dataset3Handler
    test_dataset_handler = Dataset3Handler
    
    print("Loading trained model...")
    try:
        model = load_trained_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nPreparing test dataset...")
    try:
        test_generator, test_datasets = prepare_test_dataset(
            test_data_path, test_excel_path, test_dataset_handler
        )
        num_test_samples = len(test_datasets.get_all_cases())
        print(f"Test dataset prepared with {num_test_samples} samples")
    except Exception as e:
        print(f"Error preparing test dataset: {e}")
        return
    
    print("\nEvaluating model...")
    try:
        predictions, true_labels, pred_classes, true_classes = evaluate_model(
            model, test_generator, num_test_samples
        )
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, pred_classes, 
                                  target_names=['Class 0', 'Class 1', 'Class 2']))
        
        # Calculate and print accuracy
        accuracy = np.mean(pred_classes == true_classes)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(true_classes, pred_classes)
        
        # Plot prediction confidence
        plot_prediction_confidence(predictions, true_classes)
        
        # Save results
        results_df = pd.DataFrame({
            'true_class': true_classes,
            'pred_class': pred_classes,
            'class_0_confidence': predictions[:, 0],
            'class_1_confidence': predictions[:, 1],
            'class_2_confidence': predictions[:, 2]
        })
        
        results_df.to_csv('test_results.csv', index=False)
        print("\nResults saved to 'test_results.csv'")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return


if __name__ == '__main__':
    # Set GPU configuration (match your training setup)
    gpu_use = 0
    os.environ["KERAS_BACKEND"] = "tensorflow"