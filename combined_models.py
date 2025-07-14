import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Custom focal loss function (needed for loading models)
def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * tf.pow(1. - pt, gamma)
        return -tf.reduce_mean(focal_weight * tf.math.log(pt))
    return loss

class ThreeClassCancerClassifier:
    def __init__(self, idh_model_path, codeletion_model_path):
        """
        Initialize the three-class classifier with paths to the two binary models.
        
        Args:
            idh_model_path: Path to the IDH mutation classification model
            codeletion_model_path: Path to the 1p/19q codeletion classification model
        """
        # Load models with custom objects
        custom_objects = {
            'focal_loss': focal_loss(),
            'loss': focal_loss()
        }
        
        self.idh_model = load_model(idh_model_path, custom_objects=custom_objects)
        self.codeletion_model = load_model(codeletion_model_path, custom_objects=custom_objects)
        
        # Class mapping
        self.class_names = {0: 'Glioblastoma', 1: 'Oligodendroglioma', 2: 'Astrocytoma'}
        self.class_labels = ['Glioblastoma', 'Oligodendroglioma', 'Astrocytoma']
    
    def predict_three_class(self, data):
        """
        Predict three-class cancer type based on IDH and 1p/19q predictions.
        
        Args:
            data: Input data for prediction (same format as used for binary models)
            
        Returns:
            predictions: Array of class predictions (0, 1, or 2)
            probabilities: Array of prediction probabilities for each class
        """
        # Get predictions from both models
        idh_pred = self.idh_model.predict(data)
        codeletion_pred = self.codeletion_model.predict(data)
        
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

def get_true_labels(patients_sf, patients_egd):
    """
    Generate true three-class labels from the clinical data.
    
    Args:
        patients_sf: UCSF clinical data DataFrame
        patients_egd: Rotterdam clinical data DataFrame
        
    Returns:
        Function to get true label for a given patient
    """
    def get_label(patient_file):
        patient = patient_file.split('/')[-1].split('.')[0]
        
        if 'UCSF' in patient:
            # Check if patient exists in dataframe
            patient_row = patients_sf[patients_sf.ID == patient]
            if patient_row.empty:
                return None
                
            # Get IDH status
            idh_status = patient_row.IDH.values[0]
            
            # Check for glioblastoma first (explicit diagnosis)
            tumor_type = patient_row['Final pathologic diagnosis (WHO 2021)']
            if not tumor_type.empty and 'glioblastoma' in str(tumor_type.values[0]).lower():
                return 0  # Glioblastoma
            
            if idh_status == 'wildtype':
                return 0  # Glioblastoma
            else:  # IDH mutant
                codeletion = patient_row['1p/19q']
                if codeletion.isna().values[0] or codeletion.values[0] == '':
                    return None  # Exclude uncertain cases
                if codeletion.values[0] == 'relative co-deleted':
                    return 1  # Oligodendroglioma
                else:
                    return 2  # Astrocytoma
        else:
            # Rotterdam data
            patient_row = patients_egd[patients_egd.Subject == patient]
            if patient_row.empty:
                return None
                
            idh_status = patient_row.who_idh_mutation_status.values[0]
            
            if idh_status == 0:  # IDH wildtype
                return 0  # Glioblastoma
            elif idh_status == 1:  # IDH mutant
                codeletion = patient_row['who_1p19q_codeletion'].values[0]
                if codeletion == 1:
                    return 1  # Oligodendroglioma
                elif codeletion == 0:
                    return 2  # Astrocytoma
                elif codeletion == -1:
                    # Check the 'type' column as in your test code
                    type_val = patient_row['type'].values[0]
                    if type_val == 2:
                        return 0  # treat as glioblastoma (not codeleted)
                    else:
                        return None  # exclude this patient
                else:
                    return None  # for any unexpected value
            else:
                return None
    
    return get_label

def load_data_generator(patient_files, batch_size=1):
    """
    Data generator compatible with your datagenerator module.
    """
    from datagenerator import datagen
    return datagen(patient_files, [0] * len(patient_files), batch_size=batch_size, augment=False)

def evaluate_three_class_model(model_paths, clinical_data_paths):
    """
    Evaluate the three-class classification performance using the same data structure as training.
    
    Args:
        model_paths: Dict with 'idh' and 'codeletion' model paths
        clinical_data_paths: Dict with paths to clinical data files
    """
    # Data paths from your test code
    patient_loc_sf = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/images_t1_t2_fl'
    patient_loc_egd = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl'
    
    # Load clinical data exactly like in your test code
    patients_sf = pd.read_excel(clinical_data_paths['ucsf'], engine='openpyxl')
    patients_sf['ID'] = patients_sf['ID'].str.replace(r'(\d+)$', lambda m: f"{int(m.group(1)):04}", regex=True)
    patients_sf = patients_sf[~patients_sf.ID.isna()]
    
    patients_egd = pd.read_excel(clinical_data_paths['rotterdam'])
    patients_egd = patients_egd[~patients_egd.Subject.isna()]
    
    # Combine all patients like in your training code
    patients = list(patients_sf.ID) + list(patients_egd.Subject)
    
    # Use the same train/val split to get test patients (using validation set as test)
    train_patients, test_patients = train_test_split(patients, test_size=0.1, random_state=42)
    test_patient_files = [f'{patient_loc_sf}/{patient}.npy' if 'UCSF' in patient else f'{patient_loc_egd}/{patient}.npy' for patient in test_patients]
    
    # Filter out missing files
    test_patient_files = [p for p in test_patient_files if os.path.exists(p)]
    
    # Initialize classifier
    classifier = ThreeClassCancerClassifier(
        model_paths['idh'], 
        model_paths['codeletion']
    )
    
    # Get label function
    get_label = get_true_labels(patients_sf, patients_egd)
    
    # Get true labels and filter out None labels
    test_pairs = [(p, get_label(p)) for p in test_patient_files]
    test_pairs = [(p, y) for p, y in test_pairs if y is not None]
    
    if not test_pairs:
        print("No valid test samples found!")
        return None, None, None
        
    test_files, true_labels = zip(*test_pairs)
    
    print(f"Found {len(test_files)} test samples with valid labels")
    print(f"Label distribution: {np.bincount(true_labels)}")
    
    # Load your custom data generator
    sys.path.append('/scratch/radv/ijwamelink/classification/')
    from datagenerator import datagen
    
    # Create data generator for test data
    test_gen = datagen(test_files, [0] * len(test_files), batch_size=1, augment=False)
    
    # Make predictions
    all_predictions = []
    all_probabilities = []
    
    print("Making predictions...")
    for i, (data, _) in enumerate(test_gen):
        if i >= len(test_files):  # Stop when we've processed all files
            break
            
        # Predict
        pred, prob = classifier.predict_three_class(data)
        all_predictions.append(pred[0])
        all_probabilities.append(prob[0])
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_files)} samples")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, all_predictions)
    
    print(f"\nThree-Class Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, all_predictions, 
                              target_names=classifier.class_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.class_labels, 
                yticklabels=classifier.class_labels)
    plt.title('Confusion Matrix - Three-Class Cancer Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy, all_predictions, all_probabilities

# Example usage
if __name__ == "__main__":
    # Define paths to your trained models
    model_paths = {
        'idh': '/path/to/your/idh_model.keras',
        'codeletion': '/path/to/your/codeletion_model.keras'
    }
    
    # Define paths to clinical data
    clinical_data_paths = {
        'ucsf': '/home/radv/ofilipowicz/my-scratch/datasetlabels/UCSF-PDGM_clinical_data.xlsx',
        'rotterdam': '/scratch/radv/ijwamelink/classification/Genetic_data.csv'  # or Rotterdam_clinical_data.xls
    }
    
    # Path to test data
    test_data_path = '/path/to/your/test/data'
    
    # Run evaluation
    accuracy, predictions, probabilities = evaluate_three_class_model(
        model_paths, clinical_data_paths
    )
    
    print(f"\nFinal three-class accuracy: {accuracy:.4f}")