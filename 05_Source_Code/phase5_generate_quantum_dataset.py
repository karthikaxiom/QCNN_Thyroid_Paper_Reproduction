import os
import numpy as np

from utilities.quantum_utils import quantum_filter

print("=" * 60)
print("PHASE 5 : GENERATING QUANTUM DATASET")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")

# Load datasets

X_train = np.load(os.path.join(MODEL_FOLDER, "X_train_smote.npy"))
X_test = np.load(os.path.join(MODEL_FOLDER, "X_test.npy"))

print("\nTraining Images :", len(X_train))
print("Testing Images  :", len(X_test))

# Quantum Feature Extraction Function

def quantum_transform(images):

    transformed = []

    total = len(images)

    for index, image in enumerate(images):

        print(f"Processing Image {index+1}/{total}")

        image = image.squeeze()

        feature_maps = np.zeros((64, 64, 4), dtype=np.float32)

        for i in range(64):

            for j in range(64):

                patch = image[
                    i*2:i*2+2,
                    j*2:j*2+2
                ].flatten()

                feature_maps[i, j] = quantum_filter(patch)

        transformed.append(feature_maps)

    return np.array(transformed)

# Generate Quantum Dataset

print("\nGenerating Quantum Training Dataset...")

X_train_quantum = quantum_transform(X_train)

print("\nGenerating Quantum Testing Dataset...")

X_test_quantum = quantum_transform(X_test)

# Save

np.save(
    os.path.join(MODEL_FOLDER, "X_train_quantum.npy"),
    X_train_quantum
)

np.save(
    os.path.join(MODEL_FOLDER, "X_test_quantum.npy"),
    X_test_quantum
)

print("\nSaved Successfully")

print("X_train_quantum.npy :", X_train_quantum.shape)
print("X_test_quantum.npy  :", X_test_quantum.shape)

print("=" * 60)
print("PHASE 5 STEP 3 COMPLETED")
print("=" * 60)