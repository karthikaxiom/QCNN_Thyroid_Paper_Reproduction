from pathlib import Path
from PIL import Image

# Paths

INPUT_FOLDER = Path("../02_Dataset_Raw/archive")
OUTPUT_FOLDER = Path("../03_Dataset_Processed/resized_images")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Settings

TARGET_SIZE = (128, 128)

# Find Images

image_files = sorted(INPUT_FOLDER.glob("*.jpg"))

print("=" * 50)
print("QCNN Image Resizer")
print("=" * 50)

print(f"\nImages Found : {len(image_files)}")

saved = 0

# Resize Images

for image_path in image_files:

    image = Image.open(image_path)

    resized = image.resize(
        TARGET_SIZE,
        Image.Resampling.LANCZOS
    )

    output_path = OUTPUT_FOLDER / image_path.name

    resized.save(output_path)

    saved += 1

# Verification

print("\nProcessing Complete")
print("-" * 30)

print(f"Images Processed : {saved}")
print(f"Output Folder    : {OUTPUT_FOLDER}")
print(f"Resize Dimension : {TARGET_SIZE}")

print("\nVerification")

output_images = list(OUTPUT_FOLDER.glob("*.jpg"))

print(f"Input Images  : {len(image_files)}")
print(f"Output Images : {len(output_images)}")

if len(image_files) == len(output_images):
    print("\nSUCCESS ✓")
    print("All images resized successfully.")
else:
    print("\nWARNING")
    print("Image count mismatch detected.")