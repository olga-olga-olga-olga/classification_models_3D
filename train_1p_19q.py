import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import sys
sys.path.append('/scratch/radv/ijwamelink/classification/')
import random
import numpy as np
import pandas as pd

from glob import glob
from keras.models import Model
from datagenerator import datagen
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from classification_models_3D.kkeras import Classifiers
from keras.mixed_precision import set_global_policy
from keras.mixed_precision import LossScaleOptimizer
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import tensorflow as tf
from keras import backend as K
import datetime

set_global_policy('mixed_float16')

patient_loc_sf = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/images_t1_t2_fl'
patient_loc_egd = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl'

patients_sf = pd.read_excel('/home/radv/ofilipowicz/my-scratch/datasetlabels/UCSF-PDGM_clinical_data.xlsx', engine='openpyxl')
patients_sf['ID'] = patients_sf['ID'].str.replace(r'(\d+)$', lambda m: f"{int(m.group(1)):04}", regex=True)
patients_sf = patients_sf[~patients_sf.ID.isna()]
patients_egd = pd.read_excel('/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls')
patients_egd = patients_egd[~patients_egd.Subject.isna()]


patients = list(patients_sf.ID) + list(patients_egd.Subject)


# Split into 90% train, 10% val
train_patients, val_patients = train_test_split(patients, test_size=0.1, random_state=42)
train_patients = [f'{patient_loc_sf}/{patient}.npy' if 'UCSF' in patient else f'{patient_loc_egd}/{patient}.npy' for patient in train_patients]
val_patients = [f'{patient_loc_sf}/{patient}.npy' if 'UCSF' in patient else f'{patient_loc_egd}/{patient}.npy' for patient in val_patients]

def get_output(patient):
    patient = patient.split('/')[-1].split('.')[0]
    if 'UCSF' in patient:
        tumor_type = patients_sf.loc[patients_sf.ID == patient, 'Final pathologic diagnosis (WHO 2021)']
        if not tumor_type.empty and 'glioblastoma' in str(tumor_type.values[0]).lower():
            return 0  # glioblastoma is always not codeleted
        codeletion = patients_sf.loc[patients_sf.ID == patient, '1p/19q']
        # Exclude if 1p/19q is missing or empty
        if codeletion.isna().values[0] or codeletion.values[0] == '':
            return None
        if codeletion.values[0] == 'relative co-deleted':
            return 1
        else:
            return 0
    else:
        row = patients_egd[patients_egd.Subject == patient]
        if row.empty:
            return None
        codeletion = row['who_1p19q_codeletion'].values[0]
        if codeletion in [0, 1]:
            return codeletion
        elif codeletion == -1:
            # Check the 'type' column
            type_val = row['type'].values[0]
            if type_val == 2:
                return 0  # treat as not codeleted
            else:
                return None  # exclude this patient
        else:
            return None  # for any unexpected value
    


def get_output(patient):
    patient = patient.split('/')[-1].split('.')[0]
    if 'UCSF' in patient:
        IDH = patients_sf.IDH[patients_sf.ID == patient]
        if IDH.values[0] == 'wildtype':
            return 0
        else:
            return 1
    else:
        IDH = patients_egd.who_idh_mutation_status[patients_egd.Subject == patient]
        return IDH.values[0]








# Filter out missing files
train_patients = [p for p in train_patients if os.path.exists(p)]
val_patients = [p for p in val_patients if os.path.exists(p)]

# Update labels to match filtered files and exclude None labels
train_pairs = [(p, get_output(p)) for p in train_patients]
train_pairs = [(p, y) for p, y in train_pairs if y is not None]
train_patients, train_output = zip(*train_pairs) if train_pairs else ([], [])

val_pairs = [(p, get_output(p)) for p in val_patients]
val_pairs = [(p, y) for p, y in val_pairs if y is not None]
val_patients, val_output = zip(*val_pairs) if val_pairs else ([], [])

print(f"Training with {len(train_patients)} train files, {len(val_patients)} val files")

batch_size = 1
num_classes = 1
patience = 50
learning_rate = 0.0001
model_type = 'densenet169'
epochs = 50

ResNet18, preprocess_input = Classifiers.get('densenet169')
model = ResNet18(input_shape=(240, 240, 160, 3), classes=num_classes, weights='imagenet')

x = model.layers[-1].output
x = GlobalAveragePooling3D()(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, name='prediction')(x)
x = Activation('sigmoid')(x)
model = Model(inputs=model.inputs, outputs=x)

print(model.summary())
optim = Adam(learning_rate=learning_rate)
optim = LossScaleOptimizer(optim)

def focal_loss(alpha=0.85, gamma=2.1):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())  # avoid log(0)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * tf.pow(1. - pt, gamma)
        return -tf.reduce_mean(focal_weight * tf.math.log(pt))
    return loss

loss_to_use = focal_loss(gamma=2.0, alpha=0.75)
# loss_to_use = 'binary_crossentropy'
model.compile(optimizer=optim, loss=loss_to_use, metrics=['acc', ], jit_compile=True)

# Change all model/log save paths to your directory
save_dir = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_1cat_1q19p/'

# Create a timestamped subdirectory for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_save_dir = os.path.join(save_dir, f"run_{timestamp}")
os.makedirs(run_save_dir, exist_ok=True)

cache_model_path = os.path.join(run_save_dir, '{}_temp.keras'.format(model_type))
best_model_path = os.path.join(run_save_dir, '{}-{{val_loss:.4f}}-{{epoch:02d}}.keras'.format(model_type))
csv_log_path = os.path.join(run_save_dir, 'history_{}_lr_{}.csv'.format(model_type, learning_rate))

callbacks = [
    ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
    ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
    CSVLogger(csv_log_path, append=True),
    EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
]

train_gen = datagen(train_patients, train_output, batch_size=batch_size, augment=True)
train_val = datagen(val_patients, val_output, batch_size=batch_size)

history = model.fit(
    train_gen,
    epochs=epochs, 
    validation_data=train_val,
    verbose=1,
    initial_epoch=0,
    callbacks=callbacks
)

best_loss = max(history.history['val_loss'])
print('Training finished. Loss: {}'.format(best_loss))
