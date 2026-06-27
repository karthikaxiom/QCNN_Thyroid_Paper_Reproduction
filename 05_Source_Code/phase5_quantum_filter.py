import os
import numpy as np

from utilities.quantum_utils import quantum_filter

print("=" * 60)
print("PHASE 5 : QUANTUM FILTER TRANSFORMATION")
print("=" * 60)

# Paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "06_Models"
)

RESULT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "07_Results"
)

os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load ONE sample image

X_train = np.load(
    os.path.join(MODEL_FOLDER, "X_train_smote.npy")
)

print("\nTraining Dataset Shape")

print(X_train.shape)

image = X_train[0]

print("\nInput Image Shape")

print(image.shape)

# remove channel

image = image.squeeze()

# Create Output Feature Maps

feature_maps = np.zeros(
    (64, 64, 4),
    dtype=np.float32
)

# Quantum Filtering

print("\nApplying Quantum Filter...")

for i in range(64):

    for j in range(64):

        patch = image[
            i*2:i*2+2,
            j*2:j*2+2
        ].flatten()

        output = quantum_filter(patch)

        feature_maps[i, j, :] = output

print("\nQuantum Feature Map Shape")

print(feature_maps.shape)

# Save

np.save(

    os.path.join(
        RESULT_FOLDER,
        "sample_quantum_feature_map.npy"
    ),

    feature_maps

)

print("\nSaved")

print("sample_quantum_feature_map.npy")

print("=" * 60)
print("PHASE 5 STEP 3 COMPLETE")
print("=" * 60)