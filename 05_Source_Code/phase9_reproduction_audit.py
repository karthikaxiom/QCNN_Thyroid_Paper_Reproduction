from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "06_Models"
RESULT_DIR = PROJECT_ROOT / "07_Results" / "phase9_reproduction_audit"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

X_ORIGINAL_PATH = MODEL_DIR / "X_tirads_train.npy"
Y_ORIGINAL_PATH = MODEL_DIR / "y_tirads_train.npy"

X_SMOTE_PATH = MODEL_DIR / "X_tirads_train_smote.npy"
Y_SMOTE_PATH = MODEL_DIR / "y_tirads_train_smote.npy"

X_QUANTUM_SMOTE_PATH = MODEL_DIR / "X_tirads_train_quantum.npy"

X_QUANTUM_ORIGINAL_PATH = (
    MODEL_DIR / "X_tirads_train_quantum_original.npy"
)
Y_ORIGINAL_OUTPUT_PATH = (
    MODEL_DIR / "y_tirads_train_original.npy"
)

REPORT_PATH = RESULT_DIR / "audit_report.txt"


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(missing)
        )


def sample_hash(sample: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(sample)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def class_distribution(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.astype(int), minlength=6)


def main() -> None:
    print("=" * 72)
    print("PHASE 9 : REPRODUCTION CORRECTION AUDIT")
    print("=" * 72)

    require_files(
        [
            X_ORIGINAL_PATH,
            Y_ORIGINAL_PATH,
            X_SMOTE_PATH,
            Y_SMOTE_PATH,
            X_QUANTUM_SMOTE_PATH,
        ]
    )

    x_original = np.load(X_ORIGINAL_PATH)
    y_original = np.load(Y_ORIGINAL_PATH).astype(np.int32).ravel()

    x_smote = np.load(X_SMOTE_PATH)
    y_smote = np.load(Y_SMOTE_PATH).astype(np.int32).ravel()

    x_quantum_smote = np.load(X_QUANTUM_SMOTE_PATH)

    n_original = len(x_original)

    print("\nDataset Shapes")
    print("-" * 72)
    print(f"Original images         : {x_original.shape}")
    print(f"Original labels         : {y_original.shape}")
    print(f"SMOTE images            : {x_smote.shape}")
    print(f"SMOTE labels            : {y_smote.shape}")
    print(f"SMOTE quantum features  : {x_quantum_smote.shape}")

    if len(x_original) != len(y_original):
        raise ValueError("Original image-label count mismatch.")

    if len(x_smote) != len(y_smote):
        raise ValueError("SMOTE image-label count mismatch.")

    if len(x_smote) != len(x_quantum_smote):
        raise ValueError("SMOTE and quantum-feature count mismatch.")

    image_prefix_match = np.allclose(
        x_smote[:n_original],
        x_original,
        rtol=0.0,
        atol=1e-7,
    )

    label_prefix_match = np.array_equal(
        y_smote[:n_original],
        y_original,
    )

    print("\nSMOTE Ordering Verification")
    print("-" * 72)
    print(f"Original-image prefix match : {image_prefix_match}")
    print(f"Original-label prefix match : {label_prefix_match}")

    quantum_nan_count = int(np.isnan(x_quantum_smote).sum())
    quantum_inf_count = int(np.isinf(x_quantum_smote).sum())

    sample_std = np.std(
        x_quantum_smote.reshape(len(x_quantum_smote), -1),
        axis=1,
    )

    print("\nQuantum Feature Integrity")
    print("-" * 72)
    print(f"NaN values                : {quantum_nan_count}")
    print(f"Infinite values           : {quantum_inf_count}")
    print(f"Minimum sample deviation  : {sample_std.min():.8f}")
    print(f"Maximum sample deviation  : {sample_std.max():.8f}")
    print(f"Constant feature maps     : {int(np.sum(sample_std == 0))}")

    hash_to_labels: dict[str, list[int]] = defaultdict(list)

    for feature_map, label in zip(x_quantum_smote, y_smote):
        hash_to_labels[sample_hash(feature_map)].append(int(label))

    duplicate_groups = 0
    conflicting_groups = 0
    duplicate_samples = 0

    for labels in hash_to_labels.values():
        if len(labels) > 1:
            duplicate_groups += 1
            duplicate_samples += len(labels) - 1

            if len(set(labels)) > 1:
                conflicting_groups += 1

    print("\nDuplicate Audit")
    print("-" * 72)
    print(f"Unique quantum maps       : {len(hash_to_labels)}")
    print(f"Repeated samples          : {duplicate_samples}")
    print(f"Duplicate groups          : {duplicate_groups}")
    print(f"Conflicting-label groups  : {conflicting_groups}")

    print("\nOriginal Class Distribution")
    print("-" * 72)

    for class_id, count in enumerate(class_distribution(y_original)):
        print(f"Class {class_id}: {count}")

    extraction_successful = (
        image_prefix_match
        and label_prefix_match
        and quantum_nan_count == 0
        and quantum_inf_count == 0
    )

    if extraction_successful:
        x_quantum_original = x_quantum_smote[:n_original].copy()

        np.save(
            X_QUANTUM_ORIGINAL_PATH,
            x_quantum_original.astype(np.float32),
        )

        np.save(
            Y_ORIGINAL_OUTPUT_PATH,
            y_original.astype(np.int32),
        )

        print("\nOriginal quantum dataset extracted successfully.")
        print(f"Saved: {X_QUANTUM_ORIGINAL_PATH}")
        print(f"Saved: {Y_ORIGINAL_OUTPUT_PATH}")
        print(f"Shape: {x_quantum_original.shape}")
    else:
        print(
            "\nOriginal quantum features were not extracted because "
            "the ordering or integrity checks failed."
        )

    report = f"""
PHASE 9 REPRODUCTION CORRECTION AUDIT
=====================================

Original images                 : {x_original.shape}
SMOTE images                    : {x_smote.shape}
SMOTE quantum features          : {x_quantum_smote.shape}

Original-image prefix match     : {image_prefix_match}
Original-label prefix match     : {label_prefix_match}

NaN values                      : {quantum_nan_count}
Infinite values                 : {quantum_inf_count}
Constant quantum maps           : {int(np.sum(sample_std == 0))}

Unique quantum maps             : {len(hash_to_labels)}
Repeated samples                : {duplicate_samples}
Duplicate groups                : {duplicate_groups}
Conflicting-label groups        : {conflicting_groups}

Original quantum extraction     : {extraction_successful}
"""

    REPORT_PATH.write_text(report.strip() + "\n", encoding="utf-8")

    print(f"\nAudit report saved: {REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()