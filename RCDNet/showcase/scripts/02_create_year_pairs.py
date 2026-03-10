#!/usr/bin/env python3
"""
Cross-Year Temporal Pairing Script
Creates 2023→2024 image pairs and tiles them into 512×512 patches for RCDNet.

Usage:
    python 02_create_year_pairs.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


def create_tiles(image: np.ndarray, patch_size: int = 512, overlap: int = 128):
    """
    Tile a large image into overlapping patches.

    Args:
        image: Input image [H, W, C]
        patch_size: Size of each patch
        overlap: Overlap between patches

    Returns:
        List of (patch, row, col) tuples
    """
    H, W = image.shape[:2]
    stride = patch_size - overlap
    tiles = []

    rows = (H - overlap) // stride
    cols = (W - overlap) // stride

    for i in range(rows):
        for j in range(cols):
            y = i * stride
            x = j * stride

            # Ensure we don't go out of bounds
            if y + patch_size > H:
                y = H - patch_size
            if x + patch_size > W:
                x = W - patch_size

            patch = image[y:y+patch_size, x:x+patch_size]
            tiles.append((patch, i, j))

    return tiles


def main():
    print("=" * 80)
    print("Cross-Year Temporal Pairing: 2023 → 2024")
    print("=" * 80)
    print()

    # Paths
    data_2023_dir = Path("showcase/data_2023/raw")
    data_2024_dir = Path("showcase/data/raw")
    output_dir = Path("showcase/data_comparison")

    # Create output directories
    (output_dir / "A").mkdir(parents=True, exist_ok=True)
    (output_dir / "B").mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    images_2023 = sorted(data_2023_dir.glob("T31UEQ_*.png"))
    images_2024 = sorted(data_2024_dir.glob("T31UEQ_*.png"))

    print("Available 2023 images:")
    for img in images_2023:
        date = img.stem.split("_")[1]
        print(f"  • {date}: {img.name}")

    print("\nAvailable 2024 images:")
    for img in images_2024:
        date = img.stem.split("_")[1]
        print(f"  • {date}: {img.name}")

    # Create pairs: Use June 24, 2023 as before and June 23, 2024 as after
    before_img = None
    after_img = None

    for img in images_2023:
        if "20230624" in img.name:
            before_img = img
            break

    for img in images_2024:
        if "20240623" in img.name:
            after_img = img
            break

    if not before_img or not after_img:
        print("\n❌ Error: Could not find matching pair")
        print(f"   Looking for: 2023-06-24 and 2024-06-23")
        return

    # Calculate time difference
    date_before = datetime.strptime("20230624", "%Y%m%d")
    date_after = datetime.strptime("20240623", "%Y%m%d")
    days_apart = (date_after - date_before).days

    print(f"\n✅ Found perfect year-over-year pair:")
    print(f"   Before: {before_img.name} (2023-06-24)")
    print(f"   After:  {after_img.name} (2024-06-23)")
    print(f"   Time span: {days_apart} days (~{days_apart/365:.1f} years)")
    print()

    # Load images
    print("Loading images...")
    img_before = np.array(Image.open(before_img).convert("RGB"))
    img_after = np.array(Image.open(after_img).convert("RGB"))

    print(f"  Before size: {img_before.shape}")
    print(f"  After size: {img_after.shape}")

    if img_before.shape != img_after.shape:
        print("\n❌ Error: Image shapes don't match")
        return

    # Create tiles
    print("\nTiling images (patch_size=512, overlap=128)...")
    tiles_before = create_tiles(img_before, patch_size=512, overlap=128)
    tiles_after = create_tiles(img_after, patch_size=512, overlap=128)

    print(f"  Created {len(tiles_before)} patches")

    # Save patches
    print("\nSaving patches...")
    patch_list = []

    for idx, ((patch_before, row, col), (patch_after, _, _)) in enumerate(
        tqdm(zip(tiles_before, tiles_after), total=len(tiles_before), desc="Saving patches")
    ):
        patch_id = f"pair00_patch{idx:04d}"

        # Save before (A)
        Image.fromarray(patch_before).save(output_dir / "A" / f"{patch_id}.png")

        # Save after (B)
        Image.fromarray(patch_after).save(output_dir / "B" / f"{patch_id}.png")

        # Create dummy ground truth (all zeros)
        dummy_gt = np.zeros((512, 512), dtype=np.uint8)
        Image.fromarray(dummy_gt).save(output_dir / "gt" / f"{patch_id}.png")

        patch_list.append(patch_id)

    # Save patch list
    pairs_file = output_dir / "pairs.txt"
    with open(pairs_file, 'w') as f:
        for patch_id in patch_list:
            f.write(f"{patch_id}\n")

    print(f"✅ Saved {len(patch_list)} patch pairs")

    # Save metadata
    metadata = {
        "before_date": "2023-06-24",
        "after_date": "2024-06-23",
        "days_apart": days_apart,
        "tile_id": "T31UEQ",
        "patch_count": len(patch_list),
        "patch_size": 512,
        "overlap": 128,
        "before_image": str(before_img),
        "after_image": str(after_img)
    }

    metadata_file = output_dir / "pairing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "=" * 80)
    print("PAIRING SUMMARY")
    print("=" * 80)
    print(f"Time period: 2023-06-24 → 2024-06-23")
    print(f"Time span: {days_apart} days (1 year)")
    print(f"Total patches: {len(patch_list)}")
    print(f"Patch size: 512×512")
    print(f"Overlap: 128 pixels")
    print()
    print("Output structure:")
    print(f"  A/ - Before images ({len(patch_list)} patches)")
    print(f"  B/ - After images ({len(patch_list)} patches)")
    print(f"  pairs.txt - Patch ID list")
    print(f"  pairing_metadata.json - Processing metadata")
    print()
    print("✅ Ready for RCDNet inference!")


if __name__ == "__main__":
    main()
