import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("PHASE 8 : TI-RADS DATASET PREPARATION")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(PROJECT_ROOT, "03_Dataset_Processed", "final_dataset.csv")
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "03_Dataset_Processed", "resized_images")
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")

os.makedirs(MODEL_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df = df[df["tirads"].isin(["2", "3", "4a", "4b", "4c", "5"])]

print("\nValid TI-RADS Images:", len(df))
print("\nTI-RADS Distribution")
print(df["tirads"].value_counts())

label_map = {
    "2": 0,
    "3": 1,
    "4a": 2,
    "4b": 3,
    "4c": 4,
    "5": 5
}

df["label"] = df["tirads"].map(label_map)

X = []
y = []
missing = []

for _, row in df.iterrows():

    image_path = os.path.join(IMAGE_FOLDER, row["image_name"])

    if not os.path.exists(image_path):
        missing.append(image_path)
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)

    X.append(img)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("\nImages Loaded:", X.shape)
print("Labels Loaded:", y.shape)
print("Missing Images:", len(missing))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)

np.save(os.path.join(MODEL_FOLDER, "X_tirads_train.npy"), X_train)
np.save(os.path.join(MODEL_FOLDER, "X_tirads_test.npy"), X_test)
np.save(os.path.join(MODEL_FOLDER, "y_tirads_train.npy"), y_train)
np.save(os.path.join(MODEL_FOLDER, "y_tirads_test.npy"), y_test)

print("\nSaved:")
print("X_tirads_train.npy")
print("X_tirads_test.npy")
print("y_tirads_train.npy")
print("y_tirads_test.npy")

print("=" * 60)
print("PHASE 8 STEP 1 COMPLETE")
print("=" * 60)