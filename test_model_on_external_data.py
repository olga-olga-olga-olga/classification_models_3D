import os
import numpy as np
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

if __name__ == '__main__':
    # --- CONFIGURE THESE PATHS ---
    model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/resnet18-0.1223-09.keras'  
    data_path = "/data/share/IMAGO/Rotterdam/"          
    excel_path = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"   
    num_classes = 3
    shape_size = (240, 240, 128, 3)
    steps = 100  # Set to number of batches in your test set
    batch_size = 1  

    # --- PREPARE DATASET AND GENERATOR ---
    datasets = Datasets(target_size=(240, 240, 128), target_spacing=(1.0, 1.0, 1.0))
    # Use the appropriate handler for your external data
    datasets.add_dataset(data_path, excel_path, Dataset3Handler)  # Change handler if needed

    # Only need the validation generator for testing
    _, gen_test, _ = create_batch_generators(
        datasets,
        batch_size_train=batch_size,
        batch_size_valid=batch_size
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
    results = model.evaluate(gen_test, steps=steps, verbose=1)
    print("Test results (loss, acc, class_0_acc, class_1_acc, class_2_acc):", results)

    # --- OPTIONAL: PREDICT ---
    # preds = model.predict(gen_test, steps=steps)
    # print("Predictions shape:", preds.shape)