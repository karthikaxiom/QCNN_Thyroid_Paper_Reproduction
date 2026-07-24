from pathlib import Path
import os
from PIL import Image
import numpy as np

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

# Main Inspection

def inspect_images(dataset_path):

    image_files = sorted(
        [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith(".jpg")
        ]
    )

    total_images = len(image_files)

    readable = 0
    corrupted = 0

    grayscale = 0
    rgb = 0

    widths = []
    heights = []

    pixel_min = []
    pixel_max = []
    pixel_mean = []

    print("=" * 60)
    print("QCNN PAPER REPRODUCTION")
    print("Phase 4 : Image Inspection")
    print("=" * 60)

    print(f"\nImages Found : {total_images}\n")

    for image_name in image_files:

        image_path = os.path.join(dataset_path, image_name)

        try:

            img = Image.open(image_path)

            img_array = np.array(img)

            readable += 1

            widths.append(img.width)
            heights.append(img.height)

            if len(img_array.shape) == 2:

                grayscale += 1

            elif len(img_array.shape) == 3:

                rgb += 1

            pixel_min.append(img_array.min())
            pixel_max.append(img_array.max())
            pixel_mean.append(img_array.mean())

        except Exception:

            corrupted += 1

    print("=" * 60)
    print("Inspection Summary")
    print("=" * 60)

    print(f"Readable Images : {readable}")
    print(f"Corrupted Images : {corrupted}")

    print("\nImage Channels")
    print("------------------------")

    print(f"Grayscale : {grayscale}")
    print(f"RGB       : {rgb}")

    print("\nResolution Statistics")
    print("------------------------")

    print(f"Minimum Width  : {min(widths)}")
    print(f"Maximum Width  : {max(widths)}")
    print(f"Average Width  : {sum(widths)/len(widths):.2f}")

    print()

    print(f"Minimum Height : {min(heights)}")
    print(f"Maximum Height : {max(heights)}")
    print(f"Average Height : {sum(heights)/len(heights):.2f}")

    print()

    print(f"Unique Widths  : {len(set(widths))}")
    print(f"Unique Heights : {len(set(heights))}")

    print("\nPixel Statistics")
    print("------------------------")

    print(f"Minimum Pixel Value : {min(pixel_min)}")
    print(f"Maximum Pixel Value : {max(pixel_max)}")
    print(f"Average Pixel Value : {sum(pixel_mean)/len(pixel_mean):.2f}")

    print("\nInspection Complete.")


# Main

if __name__ == "__main__":

    inspect_images(DATASET_DIR)