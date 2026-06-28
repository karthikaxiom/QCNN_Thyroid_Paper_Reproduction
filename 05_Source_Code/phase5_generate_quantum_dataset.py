import os
import numpy as np

from utilities.dataset_transformer import transform_dataset
from utilities import checkpoint

print("=" * 60)
print("PHASE 5 : QUANTUM DATASET GENERATION")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")

checkpoint.initialize(PROJECT_ROOT)

X_train = np.load(os.path.join(MODEL_FOLDER, "X_train_smote.npy"))
X_test = np.load(os.path.join(MODEL_FOLDER, "X_test.npy"))

print("\nTraining Images :", len(X_train))
print("Testing Images  :", len(X_test))

print("\nGenerating Quantum Training Dataset...")

X_train_quantum = transform_dataset(
    X_train,
    os.path.join(MODEL_FOLDER, "X_train_quantum.npy")
)

print("\nGenerating Quantum Testing Dataset...")

X_test_quantum = transform_dataset(
    X_test,
    os.path.join(MODEL_FOLDER, "X_test_quantum.npy")
)

print("\nTraining Quantum Shape :", X_train_quantum.shape)
print("Testing Quantum Shape  :", X_test_quantum.shape)

print("\nPHASE 5 STEP 5 COMPLETE")