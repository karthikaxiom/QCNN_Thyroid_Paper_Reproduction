
"""EXT-BN-001: Batch Normalization recalibration audit.

Exploratory analysis using training and validation data only.
Original checkpoint and frozen datasets remain unchanged.
"""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
MODEL = (
    ROOT / "06_Models"
    / "phase14_roi_categorical_baseline"
    / "best_roi_categorical_baseline.keras"
)
OUT = ROOT / "07_Results" / "exp_bn_001"
BATCH = 16


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_until_bn(model, images, target):
    x = tf.convert_to_tensor(images, dtype=tf.float32)

    for layer in model.layers:
        if layer is target:
            return x.numpy().astype(np.float64)

        if isinstance(layer, keras.layers.InputLayer):
            continue

        if isinstance(
            layer,
            (
                keras.layers.BatchNormalization,
                keras.layers.Dropout,
                keras.Sequential,
            ),
        ):
            x = layer(x, training=False)
        else:
            x = layer(x)

    raise ValueError("Target BN layer not found")


def recalibrate(model, X_train, bn_layers):
    audit = []

    for number, bn in enumerate(bn_layers, start=1):
        sums = np.zeros(bn.moving_mean.shape, dtype=np.float64)
        sums_sq = np.zeros(bn.moving_mean.shape, dtype=np.float64)
        count = 0

        old_variance = bn.moving_variance.numpy().copy()

        for start in range(0, len(X_train), BATCH):
            activations = forward_until_bn(
                model,
                X_train[start:start + BATCH],
                bn,
            )

            axes = tuple(range(activations.ndim - 1))
            count += int(np.prod(activations.shape[:-1]))

            sums += activations.sum(axis=axes)
            sums_sq += np.square(activations).sum(axis=axes)

        mean = sums / count
        variance = np.maximum(
            sums_sq / count - np.square(mean),
            0.0,
        )

        bn.moving_mean.assign(mean.astype(np.float32))
        bn.moving_variance.assign(variance.astype(np.float32))

        audit.append({
            "layer": bn.name,
            "number": number,
            "original_median_variance": float(
                np.median(old_variance)
            ),
            "recalibrated_median_variance": float(
                np.median(variance)
            ),
        })

    return audit


def measure(model, X, y, condition, partition):
    probabilities = model.predict(
        X,
        batch_size=BATCH,
        verbose=0,
    )

    if probabilities.shape != (len(y), 6):
        raise ValueError("Unexpected prediction shape")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions")

    predictions = probabilities.argmax(axis=1)

    entropy = -np.sum(
        probabilities * np.log(
            np.clip(probabilities, 1e-12, 1.0)
        ),
        axis=1,
    )

    result = {
        "condition": condition,
        "partition": partition,
        "samples": int(len(y)),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=list(range(6)),
                average="macro",
                zero_division=0,
            )
        ),
        "majority_baseline_accuracy": float(
            np.bincount(y, minlength=6).max() / len(y)
        ),
        "true_counts": np.bincount(
            y, minlength=6
        ).tolist(),
        "predicted_counts": np.bincount(
            predictions, minlength=6
        ).tolist(),
        "mean_max_probability": float(
            probabilities.max(axis=1).mean()
        ),
        "mean_entropy": float(entropy.mean()),
        "per_class_probability_std": (
            probabilities.std(axis=0).tolist()
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=list(range(6)),
        ).tolist(),
    }

    rows = [
        [
            condition,
            partition,
            i,
            int(y[i]),
            int(predictions[i]),
            *map(float, probabilities[i]),
        ]
        for i in range(len(y))
    ]

    return result, rows


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUT}"
        )

    paths = {
        f"{kind}_{partition}": (
            DATA / f"{kind}_roi_{partition}.npy"
        )
        for partition in ("train", "validation")
        for kind in ("X", "y")
    }

    for path in [MODEL, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_hash = sha256(MODEL)
    input_hashes = {
        name: sha256(path)
        for name, path in paths.items()
    }

    arrays = {}

    for partition in ("train", "validation"):
        X = np.load(
            paths[f"X_{partition}"]
        ).astype(np.float32)

        y = np.load(
            paths[f"y_{partition}"]
        ).astype(int).ravel()

        if X.shape != (len(y), 64, 64, 1):
            raise ValueError("Invalid image shape")

        if not np.isfinite(X).all():
            raise ValueError("Non-finite image values")

        if X.min() < 0 or X.max() > 1.00001:
            raise ValueError("Invalid pixel range")

        if (y < 0).any() or (y >= 6).any():
            raise ValueError("Invalid class indices")

        arrays[partition] = (X, y)

    model = keras.models.load_model(
        MODEL, compile=False
    )

    bn_layers = [
        layer for layer in model.layers
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        )
    ]

    if len(bn_layers) != 7:
        raise ValueError(
            f"Expected 7 BN layers; found {len(bn_layers)}"
        )

    original_stats = [
        (
            bn.moving_mean.numpy().copy(),
            bn.moving_variance.numpy().copy(),
        )
        for bn in bn_layers
    ]

    results = []
    prediction_rows = []
    bn_audit = []
    baseline_validation = None

    try:
        for partition, (X, y) in arrays.items():
            result, rows = measure(
                model, X, y, "original", partition
            )

            results.append(result)
            prediction_rows.extend(rows)

            if partition == "validation":
                baseline_validation = model.predict(
                    X,
                    batch_size=BATCH,
                    verbose=0,
                )

        bn_audit = recalibrate(
            model,
            arrays["train"][0],
            bn_layers,
        )

        for partition, (X, y) in arrays.items():
            result, rows = measure(
                model, X, y, "recalibrated", partition
            )

            results.append(result)
            prediction_rows.extend(rows)

    finally:
        for bn, (mean, variance) in zip(
            bn_layers, original_stats
        ):
            bn.moving_mean.assign(mean)
            bn.moving_variance.assign(variance)

    restored_validation = model.predict(
        arrays["validation"][0],
        batch_size=BATCH,
        verbose=0,
    )

    restoration_error = float(
        np.max(
            np.abs(
                restored_validation - baseline_validation
            )
        )
    )

    if restoration_error > 1e-6:
        raise RuntimeError(
            "In-memory checkpoint restoration failed"
        )

    if sha256(MODEL) != checkpoint_hash:
        raise RuntimeError(
            "Original checkpoint file changed"
        )

    OUT.mkdir(parents=True)

    metrics = {
        "experiment": "EXT-BN-001",
        "purpose": (
            "Training-only BN statistics diagnostic; "
            "not independent model validation"
        ),
        "partitions_used": ["train", "validation"],
        "test_data_accessed": False,
        "learned_weights_updated": False,
        "checkpoint_sha256": checkpoint_hash,
        "input_sha256": input_hashes,
        "restoration_max_difference": restoration_error,
        "bn_statistics": bn_audit,
        "evaluations": results,
    }

    with (OUT / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    with (OUT / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "condition",
            "partition",
            "sample_index",
            "true_class",
            "predicted_class",
            "probability_2",
            "probability_3",
            "probability_4a",
            "probability_4b",
            "probability_4c",
            "probability_5",
        ])

        writer.writerows(prediction_rows)

    print("\nEXT-BN-001 COMPLETED")
    print("=" * 72)

    for result in results:
        print(
            f"{result['condition']:12s} "
            f"{result['partition']:10s} "
            f"accuracy={result['accuracy']:.4f} "
            f"balanced={result['balanced_accuracy']:.4f} "
            f"macroF1={result['macro_f1']:.4f} "
            f"preds={result['predicted_counts']}"
        )

    print("\nCheckpoint restoration error:",
          restoration_error)
    print("Original checkpoint unchanged.")
    print("Test partition not accessed.")
    print("Saved:", OUT / "metrics.json")
    print("Saved:", OUT / "predictions.csv")


if __name__ == "__main__":
    main()
