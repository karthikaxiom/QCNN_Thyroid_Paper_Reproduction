import os
import json
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("PHASE 6 : EVALUATING QuCNet TUMOR CLASSIFIER")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "07_Results")

os.makedirs(RESULT_FOLDER, exist_ok=True)

model_path = os.path.join(MODEL_FOLDER, "best_qucnet.keras")
X_test_path = os.path.join(MODEL_FOLDER, "X_test_quantum.npy")
y_test_path = os.path.join(MODEL_FOLDER, "y_test.npy")
history_path = os.path.join(RESULT_FOLDER, "training_history.json")

required_files = [
    model_path,
    X_test_path,
    y_test_path
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing required file: {file}")

print("\nLoading model and test data...")

model = tf.keras.models.load_model(model_path)
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nPredicting...")

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\nFinal Metrics")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
)

print("\nClassification Report")
print(report)

metrics_path = os.path.join(RESULT_FOLDER, "metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("QuCNet Tumor Classification Metrics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")
    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\nClassification Report\n")
    f.write(report)

# Confusion Matrix Plot

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("QuCNet Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "confusion_matrix.png"))
plt.close()

# Training Plots

if os.path.exists(history_path):

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if "accuracy" in history and "val_accuracy" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_FOLDER, "accuracy_plot.png"))
        plt.close()

    if "loss" in history and "val_loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_FOLDER, "loss_plot.png"))
        plt.close()

print("\nSaved Results")
print("-" * 40)
print("metrics.txt")
print("confusion_matrix.png")
print("accuracy_plot.png")
print("loss_plot.png")

print("=" * 60)
print("PHASE 6 EVALUATION COMPLETE")
print("=" * 60)