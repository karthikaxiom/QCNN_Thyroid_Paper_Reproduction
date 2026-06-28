import os
import time
import numpy as np

from utilities.quantum_utils import quantum_filter_fast
from utilities import checkpoint
from config import CHECKPOINT_INTERVAL

GLOBAL_CACHE = {}
CACHE_HITS = 0
CACHE_MISSES = 0


def transform_image(image):
    global CACHE_HITS, CACHE_MISSES

    image = image.squeeze()
    feature_maps = np.zeros((64, 64, 4), dtype=np.float32)

    for i in range(64):
        for j in range(64):
            patch = image[i * 2:i * 2 + 2, j * 2:j * 2 + 2].flatten()
            key = (patch * 255).astype(np.uint8).tobytes()

            if key in GLOBAL_CACHE:
                feature = GLOBAL_CACHE[key]
                CACHE_HITS += 1
            else:
                feature = quantum_filter_fast(patch)
                GLOBAL_CACHE[key] = feature
                CACHE_MISSES += 1

            feature_maps[i, j] = feature

    return feature_maps


def transform_dataset(images, save_file):
    total = len(images)

    if os.path.exists(save_file):
        output = np.load(save_file)
        print(f"\nExisting file found: {save_file}")
        print("Loaded existing output:", output.shape)
    else:
        output = np.zeros((total, 64, 64, 4), dtype=np.float32)

    start = checkpoint.load_checkpoint()

    if start > 0:
        print(f"\nResuming from image {start + 1}")

    start_time = time.time()

    for index in range(start, total):
        print("-" * 60)
        print(f"Image {index + 1}/{total}")

        image_start = time.time()

        output[index] = transform_image(images[index])

        if (index + 1) % CHECKPOINT_INTERVAL == 0:
            np.save(save_file, output)
            checkpoint.save_checkpoint(index + 1)

        image_time = time.time() - image_start
        elapsed = time.time() - start_time
        remaining = total - (index + 1)
        eta = remaining * image_time

        print(f"Image Time : {image_time:.2f} sec")
        print(f"Elapsed    : {elapsed / 60:.2f} min")
        print(f"ETA        : {eta / 60:.2f} min")
        print(f"Cache Hits : {CACHE_HITS}")
        print(f"Executions : {CACHE_MISSES}")

    np.save(save_file, output)
    checkpoint.clear_checkpoint()

    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print(f"Unique Quantum Executions : {CACHE_MISSES}")
    print(f"Cache Hits               : {CACHE_HITS}")

    total_cache = CACHE_HITS + CACHE_MISSES

    if total_cache > 0:
        reduction = CACHE_HITS / total_cache * 100
        print(f"Repeated Computations Saved : {reduction:.2f}%")

    return output






    