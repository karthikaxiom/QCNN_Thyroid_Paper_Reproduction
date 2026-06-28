import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from phase6_qucnet_model import build_qucnet_tumor_model
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE

print("=" * 60)
print("PHASE 6 : TRAINING QuCNet TUMOR CLASSIFIER")
print("=" * 60)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "07_Results")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

X_train_path = os.path.join(MODEL_FOLDER, "X_train_quantum.npy")
X_test_path = os.path.join(MODEL_FOLDER, "X_test_quantum.npy")
y_train_path = os.path.join(MODEL_FOLDER, "y_train_smote.npy")
y_test_path = os.path.join(MODEL_FOLDER, "y_test.npy")

required_files = [
    X_train_path,
    X_test_path,
    y_train_path,
    y_test_path
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing required file: {file}")

print("\nLoading datasets...")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

print("\nBuilding model...")

model = build_qucnet_tumor_model(
    input_shape=(64, 64, 4),
    num_classes=2
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model_summary_path = os.path.join(RESULT_FOLDER, "model_summary.txt")

with open(model_summary_path, "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

model.summary()

best_model_path = os.path.join(MODEL_FOLDER, "best_qucnet.keras")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

print("\nTraining started...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

history_path = os.path.join(RESULT_FOLDER, "training_history.json")

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history.history, f, indent=4)

final_model_path = os.path.join(MODEL_FOLDER, "final_qucnet.keras")
model.save(final_model_path)

print("\nTraining complete.")
print("Best model saved :", best_model_path)
print("Final model saved:", final_model_path)
print("History saved    :", history_path)

print("=" * 60)
print("PHASE 6 TRAINING COMPLETE")
print("=" * 60)