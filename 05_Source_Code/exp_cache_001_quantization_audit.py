"""EXT-CACHE-001: read-only historical quantum cache-key collision audit.

The historical dataset transformer memoizes quantum-circuit outputs using
    (patch * 255).astype(np.uint8).tobytes().
Distinct floating-point input patches can share this key. This script
counts those mismatches in the historical *training-only* arrays; it never
calls or modifies the quantum model, reads a test set, or overwrites outputs.

Run from repository root:
    python .\05_Source_Code\exp_cache_001_quantization_audit.py

Arrays that do not exist locally are reported as missing; at least one
training file must exist. No synthetic data are reported as empirical data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "06_Models"
OUT_DIR = ROOT / "07_Results" / "exp_cache_001"

CANDIDATES = [
    "X_train.npy",
    "X_train_smote.npy",
    "X_tirads_train.npy",
    "X_tirads_train_smote.npy",
]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_array(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim not in {3, 4} or array.shape[1:3] != (128, 128):
        raise ValueError(f"Expected 128x128 grayscale images: {path}, got {array.shape}")
    if array.ndim == 4 and array.shape[3] != 1:
        raise ValueError(f"Expected one grayscale channel: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Non-numeric source")
    first_exact_for_key = {}
    keys_with_collision = set()
    mismatch_occurrences = 0
    total_patches = 0
    first_examples = []
    min_input = float("inf")
    max_input = float("-inf")
    for index in range(len(array)):
        img = np.asarray(array[index]).squeeze()
        if not np.isfinite(img).all():
            raise ValueError(f"Nonfinite pixels at image {index} in {path}")
        min_input = min(min_input, float(np.min(img)))
        max_input = max(max_input, float(np.max(img)))
        if img.min() < 0 or img.max() > 1:
            raise ValueError(f"Expected 0..1 normalized input in {path}")
        patches = img.reshape(64, 2, 64, 2).transpose(
            0, 2, 1, 3).reshape(4096, 4)
        # Match the historic quantum transformer key operation verbatim.
        quantized = (patches * 255).astype(np.uint8)
        for local_idx, (row, key_row) in enumerate(zip(patches, quantized)):
            key = key_row.tobytes()
            # Numeric-value identity, independent of input float32/float64 storage dtype.
            exact = np.asarray(row, dtype=np.float64).tobytes()
            previous = first_exact_for_key.get(key)
            if previous is None:
                first_exact_for_key[key] = exact
            elif previous != exact:
                mismatch_occurrences += 1
                keys_with_collision.add(key)
                if len(first_examples) < 8:
                    first_examples.append({
                        "image_index": index,
                        "patch_index": local_idx,
                        "quantized_cache_key_hex": key.hex(),
                        "first_seen_patch": np.frombuffer(
                            previous, dtype=np.float64).tolist(),
                        "different_patch_with_same_key": np.asarray(
                            row, dtype=np.float64).tolist(),
                    })
            total_patches += 1
        if (index + 1) % 100 == 0:
            print(f"  {path.name}: {index + 1}/{len(array)} images examined")

    return {
        "source_path": str(path.relative_to(ROOT).as_posix()),
        "source_sha256": sha256_file(path),
        "image_count": int(len(array)),
        "image_shape": list(array.shape),
        "dtype": str(array.dtype),
        "value_range": [min_input, max_input],
        "patch_count": total_patches,
        "distinct_cache_keys": len(first_exact_for_key),
        "cache_keys_with_distinct_float_inputs": len(keys_with_collision),
        "later_patches_differing_from_first_cached_patch": mismatch_occurrences,
        "collision_fraction_of_patches": mismatch_occurrences / total_patches,
        "collision_examples": first_examples,
    }


def main():
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite prior cache audit: {OUT_DIR}")
    datasets = []
    missing = []
    for name in CANDIDATES:
        path = MODEL_DIR / name
        if not path.is_file():
            missing.append(name)
            print("Not found (skipped):", name)
            continue
        print("Auditing", path.name)
        datasets.append(audit_array(path))
    if not datasets:
        raise FileNotFoundError(
            "No historical training arrays found in 06_Models/. "
            "Do not substitute validation or test arrays.")
    record = {
        "experiment": "EXT-CACHE-001",
        "scope": "Historical training input arrays only",
        "model_training_or_quantum_inference": False,
        "validation_or_test_arrays_accessed": False,
        "source_arrays_modified": False,
        "cache_key_formula": "(patch * 255).astype(np.uint8).tobytes()",
        "interpretation": (
            "A positive collision count proves lossy memoization keys "
            "for this input, not the magnitude of resulting quantum output "
            "differences or an effect on final model accuracy. Zero counts "
            "on available arrays do not prove all historic datasets were collision-free."
        ),
        "missing_optional_arrays": missing,
        "datasets": datasets,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    file = OUT_DIR / "quantum_cache_key_audit.json"
    file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("\nEXT-CACHE-001 COMPLETE")
    for row in datasets:
        print(
            f"{Path(row['source_path']).name}: "
            f"{row['cache_keys_with_distinct_float_inputs']} colliding keys; "
            f"{row['later_patches_differing_from_first_cached_patch']} "
            f"mismatched patch occurrences / {row['patch_count']} patches")
    print("Saved:", file)
    print("No model, raw image, validation/test or historic result modified.")


if __name__ == "__main__":
    main()
