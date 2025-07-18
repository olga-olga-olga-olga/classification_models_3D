import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from datagenerator import datagen
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from glob import glob

def get_1p19q_labels(df):
    labels = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        codeletion = row.get('1p/19q', None)
        tumor_type = row.get('Tumor_type', None)
        
        if pd.notna(codeletion) and codeletion != '':
            codeletion_str = str(codeletion).upper().strip()
            if 'CODELETED' in codeletion_str and 'NOT' not in codeletion_str:
                labels.append(1)
                valid_indices.append(idx)
            elif 'NOT CODELETED' in codeletion_str:
                labels.append(0)
                valid_indices.append(idx)
        elif pd.notna(tumor_type) and 'GBM' in str(tumor_type).upper():
            labels.append(0)
            valid_indices.append(idx)
    
    return labels, valid_indices

# PATHS
model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models_1cat_1q19p/run_20250717_102550/resnet50-0.2458-13.keras'
data_path = '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl/'
excel_path = '/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xlsx'

# Load data
df = pd.read_excel(excel_path, engine='openpyxl')
test_labels, valid_indices = get_1p19q_labels(df)
df_valid = df.iloc[valid_indices]
test_files = [f'{data_path}/{pid}.npy' for pid in df_valid['Pseudo']]

# Filter existing files
existing_files = []
existing_labels = []
for f, l in zip(test_files, test_labels):
    if os.path.exists(f):
        existing_files.append(f)
        existing_labels.append(l)

print(f"Testing {len(existing_files)} files")
print(f"Class distribution: {np.bincount(existing_labels)}")
print(f"Class ratio: {np.mean(existing_labels):.3f}")

# Load model and test
model = load_model(model_path, compile=False)
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

print(f"Results on {len(existing_labels)} samples:")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Sensitivity (TPR): {sensitivity:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")
print(f"PPV (Precision): {ppv:.3f}")
print(f"NPV: {npv:.3f}")
print(f"FPR: {fp/(fp+tn):.3f}")
print(f"FNR: {fn/(fn+tp):.3f}")

print(classification_report(existing_labels, pred_binary, target_names=['1p/19q-intact', '1p/19q-codeleted']))

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
ax3.set_xticklabels(['intact', 'codeleted'])
ax3.set_yticklabels(['intact', 'codeleted'])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", 
                color="white" if cm[i, j] > cm.max() / 2. else "black")

# Prediction Distribution
ax4.hist(predictions[np.array(existing_labels) == 0], bins=20, alpha=0.7, label='1p/19q-intact', color='blue')
ax4.hist(predictions[np.array(existing_labels) == 1], bins=20, alpha=0.7, label='1p/19q-codeleted', color='red')
ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Count')
ax4.set_title('Prediction Distribution')
ax4.legend()

# Get model name without extension for plot filename
model_name = os.path.splitext(os.path.basename(model_path))[0]
plot_path = f'/home/radv/ofilipowicz/my-scratch/test_results/test_results_plots_{model_name}.png'

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlots saved to: {plot_path}")

# Save metrics to a text file with the model name
metrics_path = f'/home/radv/ofilipowicz/my-scratch/test_results/test_results_metrics_{model_name}.txt'
with open(metrics_path, 'w') as f:
    f.write(f"Results on {len(existing_labels)} samples:\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"AUC: {auc:.3f}\n")
    f.write(f"Sensitivity (TPR): {sensitivity:.3f}\n")
    f.write(f"Specificity (TNR): {specificity:.3f}\n")
    f.write(f"PPV (Precision): {ppv:.3f}\n")
    f.write(f"NPV: {npv:.3f}\n")
    f.write(f"FPR: {fp/(fp+tn):.3f}\n")
    f.write(f"FNR: {fn/(fn+tp):.3f}\n\n")
    f.write(classification_report(existing_labels, pred_binary, target_names=['1p/19q-intact', '1p/19q-codeleted']))

print(f"Metrics saved to: {metrics_path}")