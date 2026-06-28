import os
import numpy as np
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("PHASE 8 : TI-RADS SMOTE BALANCING")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")

X_train = np.load(os.path.join(MODEL_FOLDER, "X_tirads_train.npy"))
y_train = np.load(os.path.join(MODEL_FOLDER, "y_tirads_train.npy"))

print("\nBefore SMOTE")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

unique, counts = np.unique(y_train, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

X_flat = X_train.reshape(X_train.shape[0], -1)

smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X_flat, y_train)

X_smote = X_smote.reshape(-1, 128, 128, 1)

print("\nAfter SMOTE")
print("X_smote:", X_smote.shape)
print("y_smote:", y_smote.shape)

unique, counts = np.unique(y_smote, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

np.save(os.path.join(MODEL_FOLDER, "X_tirads_train_smote.npy"), X_smote)
np.save(os.path.join(MODEL_FOLDER, "y_tirads_train_smote.npy"), y_smote)

print("\nSaved:")
print("X_tirads_train_smote.npy")
print("y_tirads_train_smote.npy")

print("=" * 60)
print("PHASE 8 STEP 2 COMPLETE")
print("=" * 60)