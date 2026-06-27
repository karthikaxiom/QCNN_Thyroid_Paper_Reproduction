from pathlib import Path
from PIL import Image

print("=" * 60)
print("PHASE 4 : IMAGE RESIZING")
print("=" * 60)

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "02_Dataset_Raw" / "archive"
OUTPUT_FOLDER = PROJECT_ROOT / "03_Dataset_Processed" / "resized_images"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (128, 128)

# Find images

image_files = sorted(INPUT_FOLDER.glob("*.jpg"))

print(f"\nInput Folder : {INPUT_FOLDER}")
print(f"Output Folder: {OUTPUT_FOLDER}")

print(f"\nImages Found : {len(image_files)}")

if len(image_files) == 0:
    raise Exception("No JPG images found!")

saved = 0

# Resize

for image_path in image_files:

    img = Image.open(image_path)

    img = img.convert("L")          # grayscale

    img = img.resize(
        TARGET_SIZE,
        Image.Resampling.LANCZOS
    )

    output_path = OUTPUT_FOLDER / image_path.name

    img.save(output_path)

    saved += 1

# Verify

output_images = list(OUTPUT_FOLDER.glob("*.jpg"))

print("\nProcessing Complete")
print("-" * 40)

print("Images Processed :", saved)
print("Input Images     :", len(image_files))
print("Output Images    :", len(output_images))

if len(output_images) != len(image_files):
    raise Exception("Resize failed.")

print("\nSUCCESS")
print("All images resized correctly.")