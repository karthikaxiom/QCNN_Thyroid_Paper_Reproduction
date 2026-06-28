import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE

print("=" * 60)
print("PHASE 7 : CLASSICAL CNN BASELINE")
print("=" * 60)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "07_Results")

X_train = np.load(os.path.join(MODEL_FOLDER, "X_train_smote.npy"))
X_test = np.load(os.path.join(MODEL_FOLDER, "X_test.npy"))
y_train = np.load(os.path.join(MODEL_FOLDER, "y_train_smote.npy"))
y_test = np.load(os.path.join(MODEL_FOLDER, "y_test.npy"))

y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

best_path = os.path.join(MODEL_FOLDER, "best_classical_cnn.keras")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        best_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)

print("\nCLASSICAL BASELINE RESULT")
print("-" * 40)
print(f"Test Loss    : {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

with open(os.path.join(RESULT_FOLDER, "phase7_classical_baseline_history.json"), "w") as f:
    json.dump(history.history, f, indent=4)

print("=" * 60)
print("PHASE 7 CLASSICAL BASELINE COMPLETE")
print("=" * 60)