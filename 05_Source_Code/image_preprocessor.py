from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd

# Setup your exact D: drive project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = str(PROJECT_ROOT / "03_Dataset_Processed" / "master_manifest.csv")
OUTPUT_NPZ = str(PROJECT_ROOT / "03_Dataset_Processed" / "preprocessed_matrices.npz")

def run_image_preprocessing():
    if not os.path.exists(MANIFEST_CSV):
        print("❌ Error: master_manifest.csv not found! Run the parser script first.")
        return
        
    df = pd.read_csv(MANIFEST_CSV)
    X_images = []
    y_tumor = []
    y_tirads = []
    
    # Paper category mappings to integers
    tumor_mapping = {'benign': 0, 'malignant': 1}
    tirads_mapping = {'2': 0, '3': 1, '4a': 2, '4b': 3, '4c': 4, '5': 5}
    
    print(f" Loading {len(df)} images: Resizing to 128x128 & scaling pixels...")
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        
        # Read in grayscale because ultrasound textures are structural
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 1. Resize to 128x128 pixels exactly like the paper constraints
        img_resized = cv2.resize(img, (128, 128))
        
        # 2. Convert to float array and normalize to [0.0, 1.0] range
        img_scaled = img_resized.astype(np.float32) / 255.0
        
        X_images.append(img_scaled)
        y_tumor.append(tumor_mapping[row['tumor_class']])
        y_tirads.append(tirads_mapping[str(row['tirads'])])
        
    # Convert lists to solid NumPy array blocks
    X = np.array(X_images, dtype=np.float32)
    Y_tum = np.array(y_tumor, dtype=np.int32)
    Y_tir = np.array(y_tirads, dtype=np.int32)
    
    # Expand channel dimension so it fits Conv2D layers perfectly -> (Samples, 128, 128, 1)
    X = np.expand_dims(X, axis=-1)
    
    # Save the arrays locally so they don't get erased if Colab or runtime breaks
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, X=X, y_tumor=Y_tum, y_tirads=Y_tir)
    
    print("\n=== PHASE 4: PREPROCESSING MARSHALING COMPLETE ===")
    print(f" Compressed NumPy block saved to: {OUTPUT_NPZ}")
    print(f" Image Array Shape : {X.shape} (Samples, Height, Width, Channel)")
    print(f" Tumor Labels Shape: {Y_tum.shape}")
    print(f" TI-RADS Labels Shape: {Y_tir.shape}")

if __name__ == "__main__":
    run_image_preprocessing()