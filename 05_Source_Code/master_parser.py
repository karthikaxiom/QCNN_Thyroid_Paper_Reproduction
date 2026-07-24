from pathlib import Path
import os
import xml.etree.ElementTree as ET
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

OUTPUT_DIR = str(PROJECT_ROOT / "03_Dataset_Processed")

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "master_dataset.csv"
)

OUTPUT_XLSX = os.path.join(
    OUTPUT_DIR,
    "master_dataset.xlsx"
)

# ==========================================================
# Parse Dataset
# ==========================================================

def parse_dataset(dataset_path):

    records = []

    all_files = os.listdir(dataset_path)

    xml_files = sorted(
        [f for f in all_files if f.lower().endswith(".xml")]
    )

    print("=" * 60)
    print("QCNN PAPER REPRODUCTION")
    print("Phase 3 : Building Master Dataset")
    print("=" * 60)

    print(f"\nXML Files Found : {len(xml_files)}")

    # ------------------------------------------------------

    for xml_file in xml_files:

        xml_path = os.path.join(dataset_path, xml_file)

        case_id = os.path.splitext(xml_file)[0]

        try:

            tree = ET.parse(xml_path)
            root = tree.getroot()

            age = root.findtext("age", default="")
            sex = root.findtext("sex", default="")
            composition = root.findtext("composition", default="")
            echogenicity = root.findtext("echogenicity", default="")
            margins = root.findtext("margins", default="")
            calcifications = root.findtext("calcifications", default="")
            tirads = root.findtext("tirads", default="")

            image_found = False

            for file in sorted(all_files):

                if not file.lower().endswith(".jpg"):
                    continue

                image_name = os.path.splitext(file)[0]

                if (
                    image_name == case_id
                    or image_name.startswith(case_id + "_")
                ):

                    image_found = True

                    image_number = "1"

                    if "_" in image_name:
                        image_number = image_name.split("_")[1]

                    records.append({

                        "case_id": case_id,

                        "xml_file": xml_file,

                        "image_name": file,

                        "image_number": image_number,

                        "image_path": os.path.join(
                            dataset_path,
                            file
                        ),

                        "age": age,

                        "sex": sex,

                        "composition": composition,

                        "echogenicity": echogenicity,

                        "margins": margins,

                        "calcifications": calcifications,

                        "tirads": tirads

                    })

            if not image_found:

                print(f"WARNING : No image found for {xml_file}")

        except Exception as e:

            print(f"ERROR reading {xml_file}")

            print(e)

    df = pd.DataFrame(records)

    if not df.empty:

        df = df.sort_values(
            by=["case_id", "image_name"]
        ).reset_index(drop=True)

    return df


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    if not os.path.exists(DATASET_DIR):

        print("Dataset folder not found.")

        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_df = parse_dataset(DATASET_DIR)

    # Save CSV
    master_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # Save Excel
    master_df.to_excel(
        OUTPUT_XLSX,
        index=False
    )

    print("\n" + "=" * 60)
    print("MASTER DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTotal Image Records : {len(master_df)}")

    print("\nFirst Five Records:\n")
    print(master_df.head())

    print("\nFiles Saved Successfully:\n")
    print(f"CSV   : {OUTPUT_CSV}")
    print(f"Excel : {OUTPUT_XLSX}")

    print("\nTI-RADS Distribution:\n")

    if not master_df.empty:
        print(master_df["tirads"].value_counts(dropna=False))
    else:
        print("No records found.")

    print("\nPhase 3 Completed Successfully.")