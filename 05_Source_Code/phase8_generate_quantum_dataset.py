import os
import numpy as np
from utilities import checkpoint

from utilities.dataset_transformer import transform_dataset

print("=" * 60)
print("PHASE 8 : TI-RADS QUANTUM DATASET GENERATION")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
checkpoint.initialize(PROJECT_ROOT)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "06_Models"
)

# ----------------------------
# Load datasets
# ----------------------------

X_train = np.load(
    os.path.join(
        MODEL_FOLDER,
        "X_tirads_train_smote.npy"
    )
)

X_test = np.load(
    os.path.join(
        MODEL_FOLDER,
        "X_tirads_test.npy"
    )
)

print("\nTraining Images :", len(X_train))
print("Testing Images  :", len(X_test))

# ----------------------------
# Quantum Transform
# ----------------------------

print("\nGenerating Quantum Training Dataset...")

X_train_quantum = transform_dataset(

    X_train,

    os.path.join(
        MODEL_FOLDER,
        "X_tirads_train_quantum.npy"
    )

)

print("\nGenerating Quantum Testing Dataset...")

X_test_quantum = transform_dataset(

    X_test,

    os.path.join(
        MODEL_FOLDER,
        "X_tirads_test_quantum.npy"
    )

)

print("\nTraining Quantum Shape :", X_train_quantum.shape)
print("Testing Quantum Shape  :", X_test_quantum.shape)

print("=" * 60)
print("PHASE 8 STEP 3 COMPLETE")
print("=" * 60)