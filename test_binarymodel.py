import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from datagenerator import datagen
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve

def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())  # avoid log(0)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * tf.pow(1. - pt, gamma)
        return -K.mean(focal_weight * tf.math.log(pt))
    return loss

def binary_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) \
               - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return focal_loss



# UPDATE THESE PATHS
model_path = '/home/radv/ofilipowicz/my-scratch/all_the_runs_m2/models/resnet34-0.0936-22.keras'
data_path = '/data/share/IMAGO/400_cases/IMAGO_501/bids/IMAGO_first_release/derivatives - registered/'
excel_path = '/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xlsx'

# Load data
df = pd.read_excel(excel_path)
test_files = [f'{data_path}/{pid}.npy' for pid in df['Pseudo']]  # Change column name
test_labels = df['IDH'].tolist()  # Change to your label column (0/1)

# Filter existing files
existing_files = [f for f in test_files if os.path.exists(f)]
existing_labels = [test_labels[i] for i, f in enumerate(test_files) if os.path.exists(f)]

print(f"Testing {len(existing_files)} files")

# Load model and test
model = load_model(model_path, custom_objects={'focal_loss': binary_focal_loss()})
test_gen = datagen(existing_files, existing_labels, batch_size=1)

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

print(classification_report(existing_labels, pred_binary, target_names=['IDH-wildtype', 'IDH-mutant']))

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
ax3.set_xticklabels(['IDH-wt', 'IDH-mut'])
ax3.set_yticklabels(['IDH-wt', 'IDH-mut'])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

# Prediction Distribution
ax4.hist(predictions[np.array(existing_labels) == 0], bins=20, alpha=0.7, label='IDH-wildtype', color='blue')
ax4.hist(predictions[np.array(existing_labels) == 1], bins=20, alpha=0.7, label='IDH-mutant', color='red')
ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Count')
ax4.set_title('Prediction Distribution')
ax4.legend()

plt.tight_layout()
plt.savefig('/home/radv/ofilipowicz/my-scratch/test_results_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlots saved to: /home/radv/ofilipowicz/my-scratch/test_results_plots.png")