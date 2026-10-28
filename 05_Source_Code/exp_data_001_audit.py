from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "03_Dataset_Processed"
ROI = PROCESSED / "phase13_roi_dataset"
OUT = ROOT / "07_Results" / "exp_data_001"

if OUT.exists():
    raise FileExistsError("Audit results already exist; refusing overwrite.")

master_path = PROCESSED / "master_dataset.csv"
final_path = PROCESSED / "final_dataset.csv"

master = pd.read_csv(master_path)
final = pd.read_csv(final_path)

mapping = {
    "2": "Benign",
    "3": "Benign",
    "4a": "Benign",
    "4b": "Malignant",
    "4c": "Malignant",
    "5": "Malignant",
}

tirads = final["tirads"].fillna("").astype(str).str.strip().str.lower()
diagnosis = final["diagnosis"].fillna("").astype(str).str.strip()

valid = tirads.isin(mapping)
expected = tirads.map(mapping).fillna("")

mismatches = int((diagnosis[valid] != expected[valid]).sum())

case_labels = pd.DataFrame({
    "case_id": final["case_id"].astype(str),
    "tirads": tirads,
})
case_labels = case_labels.loc[valid]

conflicts = int(
    (case_labels.groupby("case_id")["tirads"].nunique() > 1).sum()
)

files = {
    "train": ROI / "train_roi_metadata.csv",
    "validation": ROI / "validation_roi_metadata.csv",
    "test": ROI / "test_roi_metadata.csv",
}

splits = {}
for name, path in files.items():
    # Only IDs are used; no test labels or images.
    frame = pd.read_csv(path, usecols=["case_id", "roi_id"])
    cases = set(frame["case_id"].astype(str))
    roi_ids = set(frame["roi_id"].astype(str))

    if len(roi_ids) != len(frame):
        raise ValueError(f"Duplicate ROI IDs in {name}")

    splits[name] = {
        "cases": cases,
        "roi_ids": roi_ids,
        "roi_count": len(frame),
    }

overlap = {
    "train_validation": len(
        splits["train"]["cases"] & splits["validation"]["cases"]
    ),
    "train_test": len(
        splits["train"]["cases"] & splits["test"]["cases"]
    ),
    "validation_test": len(
        splits["validation"]["cases"] & splits["test"]["cases"]
    ),
}

report = {
    "experiment": "EXT-DATA-001",
    "master_rows": len(master),
    "final_rows": len(final),
    "case_ids": int(final["case_id"].nunique()),
    "labeled_images": int(valid.sum()),
    "unlabeled_images": int((~valid).sum()),
    "labeled_cases": int(case_labels["case_id"].nunique()),
    "binary_proxy_mismatches": mismatches,
    "unlabeled_records_with_diagnosis": int(
        ((~valid) & diagnosis.ne("")).sum()
    ),
    "conflicting_case_tirads_labels": conflicts,
    "split_case_overlap": overlap,
    "split_sizes": {
        name: {
            "cases": len(value["cases"]),
            "rois": value["roi_count"],
        }
        for name, value in splits.items()
    },
    "label_provenance": "TI-RADS-derived binary proxy; not pathology",
    "patient_identity_independence_verified": False,
    "test_image_or_prediction_accessed": False,
    "input_hashes": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [master_path, final_path]
    },
}

assert len(master) == len(final) == 480
assert report["labeled_images"] == 349
assert report["binary_proxy_mismatches"] == 0
assert report["conflicting_case_tirads_labels"] == 0
assert all(value == 0 for value in overlap.values())

OUT.mkdir(parents=True, exist_ok=False)
(OUT / "audit.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("\nEXT-DATA-001 COMPLETE")
for key, value in report.items():
    if key != "input_hashes":
        print(f"{key}: {value}")
print("Saved:", OUT / "audit.json")
print("No original files modified.")
