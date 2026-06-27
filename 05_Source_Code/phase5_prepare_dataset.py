import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("PHASE 5 : DATASET PREPARATION")
print("=" * 60)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "03_Dataset_Processed",
    "final_dataset.csv"
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "03_Dataset_Processed",
    "resized_images"
)

# --------------------------------------------------
# Load CSV
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("\nTotal Images :", len(df))

# Keep only labelled images
df = df[df["diagnosis"].isin(["Benign", "Malignant"])].copy()

print("Labelled Images :", len(df))

print("\nClass Distribution")
print(df["diagnosis"].value_counts())

# --------------------------------------------------
# Encode Labels
# --------------------------------------------------

label_map = {
    "Benign": 0,
    "Malignant": 1
}

df["label"] = df["diagnosis"].map(label_map)

# --------------------------------------------------
# Load Images
# --------------------------------------------------

X = []
y = []

missing = []

for _, row in df.iterrows():

    image_path = os.path.join(
        IMAGE_FOLDER,
        row["image_name"]
    )

    if not os.path.exists(image_path):
        missing.append(image_path)
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        missing.append(image_path)
        continue

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=-1)

    X.append(img)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("\nImages Loaded :", len(X))
print("Labels Loaded :", len(y))
print("Image Shape :", X.shape)

print("\nMissing Images :", len(missing))

if len(missing) > 0:
    print("\nFirst 10 Missing Files:")
    for f in missing[:10]:
        print(f)

if len(X) == 0:
    raise Exception(
        "\nERROR: No images were loaded.\n"
        "Check IMAGE_FOLDER or image filenames."
    )

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Images :", len(X_train))
print("Testing Images :", len(X_test))

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT = os.path.join(
    PROJECT_ROOT,
    "06_Models"
)

os.makedirs(OUTPUT, exist_ok=True)

np.save(os.path.join(OUTPUT, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT, "X_test.npy"), X_test)

np.save(os.path.join(OUTPUT, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT, "y_test.npy"), y_test)

print("\nDatasets saved successfully.")

print("=" * 60)
print("PHASE 5 STEP 1 COMPLETE")
print("=" * 60)