# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import sys
    import os
    # For this test, make sure that only the tested framework is available
    # sys.modules['jax'] = None
    # sys.modules['torch'] = None

    gpu_use = 0
    print(f"GPU use: {gpu_use}")
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_use}"


from classification_models_3D.kkeras import Classifiers
import numpy as np
import random
import matplotlib.pyplot as plt
# from skimage import measure
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.models import Model
from keras.src.utils import summary_utils
from dataloader import create_batch_generators, get_preprocess_input_dummy
from dataset import Datasets, Dataset1Handler, Dataset2Handler, Dataset3Handler
import tensorflow as tf
import datetime



def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model' or layer_type == 'Functional':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = summary_utils.count_params(model.trainable_weights)
    non_trainable_count = summary_utils.count_params(model.non_trainable_weights)

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes







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


def train_model_example():
    use_weights = 'imagenet'
    shape_size = (240, 240, 189, 3)  # <-- Set your desired shape here
    backbone = 'resnet18'
    num_classes = 3
    batch_size_train = 1
    batch_size_valid = 1
    learning_rate = 0.0001
    patience = 15
    epochs = 50
    steps_per_epoch = 100
    validation_steps = 20
    dropout_val = 0.3

    # --- SHAPE VARIABLES ---
    target_size = (240, 240, 189)    # <-- Set your desired target size here
    num_channels = shape_size[3]
    target_spacing = (1.0, 1.0, 1.0)

    data_path_1 = "/data/share/IMAGO/SF/"
    excel_path_1 =  "/home/radv/ofilipowicz/my-scratch/datasetlabels/UCSF-PDGM_clinical_data_adjusted.xls"
    data_path_2 = "/data/share/IMAGO/400_cases/IMAGO_501/bids/IMAGO_first_release/derivatives - registered/"
    excel_path_2 = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xls"
    data_path_3 = "/data/share/IMAGO/Rotterdam/"
    excel_path_3 = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"

    # Initialize datasets
    datasets = Datasets(
        target_size=target_size,
        target_spacing=target_spacing,
        num_channels=num_channels
    )
    datasets.add_dataset(data_path_1, excel_path_1, Dataset1Handler)
    datasets.add_dataset(data_path_2, excel_path_2, Dataset2Handler)
    datasets.add_dataset(data_path_3, excel_path_3, Dataset3Handler)
    
    # Create batch generators (replaces gen_random_volume)
    gen_train, gen_valid, class_weights = create_batch_generators(
        datasets,
        batch_size_train=batch_size_train,
        batch_size_valid=batch_size_valid,
        target_size=target_size,
        target_spacing=target_spacing,
        num_channels=num_channels
    )
    # Get model and dummy preprocess function
    modelPoint, _ = Classifiers.get(backbone)
    preprocess_input = get_preprocess_input_dummy()

    model = modelPoint(
        input_shape=shape_size,
        include_top=False,
        weights=use_weights,
    )
    x = model.layers[-1].output
    x = GlobalAveragePooling3D()(x)
    x = Dropout(dropout_val)(x)
    x = Dense(num_classes, name='prediction')(x)
    x = Activation('softmax')(x)
    model = Model(inputs=model.inputs, outputs=x)

    print(model.summary())
    print(get_model_memory_usage(batch_size_train, model))
    optim = Adam(learning_rate=learning_rate)

    alpha_weights = [1.0, 3.0, 0.6]  

    loss_to_use = categorical_focal_loss(gamma=2.0, alpha=alpha_weights)
    # loss_to_use ='categorical_crossentropy' 
    model.compile(
        optimizer=optim,
        loss=loss_to_use,
        metrics=[
            'acc',
            class_0_accuracy,
            class_1_accuracy,
            class_2_accuracy
        ]
    )

  # Change all model/log save paths to your directory
    save_dir = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_3cat/'

    # Create a timestamped subdirectory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)

    cache_model_path = os.path.join(run_save_dir, '{}_temp.keras'.format(backbone))
    best_model_path = os.path.join(run_save_dir, '{}-{{val_loss:.4f}}-{{epoch:02d}}.keras'.format(backbone))
    csv_log_path = os.path.join(run_save_dir, 'history_{}_lr_{}.csv'.format(backbone, learning_rate))

    callbacks = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
        CSVLogger(csv_log_path, append=True),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
    ]


    history = model.fit(
        gen_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=gen_valid,
        validation_steps=validation_steps,
        verbose=1,
        initial_epoch=0,
        callbacks=callbacks
    )

    best_loss = max(history.history['val_loss'])
    print('Training finished. Loss: {}'.format(best_loss))


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
    train_model_example()
