#!/usr/bin/env python3
"""
Temporal Pairing and Tiling Script
Creates temporal image pairs and tiles them into 512×512 patches for RCDNet.

Usage:
    python 02_create_pairs.py --input showcase/data/raw --output showcase/data
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_product_metadata(raw_dir: Path) -> List[Dict]:
    """
    Load all product metadata files.

    Args:
        raw_dir: Directory containing extracted RGB composites

    Returns:
        List of metadata dictionaries sorted by date
    """
    metadata_files = sorted(raw_dir.glob("*_metadata.json"))
    products = []

    for meta_file in metadata_files:
        with open(meta_file) as f:
            products.append(json.load(f))

    # Sort by acquisition date
    products.sort(key=lambda x: x['acquisition_date'])

    return products


def find_temporal_pairs(
    products: List[Dict],
    max_days_apart: int = 30
) -> List[Dict]:
    """
    Find temporal pairs (before/after) from products.

    Strategy:
        1. Group by tile_id
        2. For each tile, find consecutive acquisitions
        3. Create pairs with earlier as 'before', later as 'after'

    Args:
        products: List of product metadata
        max_days_apart: Maximum days between before/after images

    Returns:
        List of pair dictionaries
    """
    # Group products by tile ID
    by_tile = defaultdict(list)
    for product in products:
        tile_id = product['tile_id']
        by_tile[tile_id].append(product)

    pairs = []
    pair_id = 0

    for tile_id, tile_products in by_tile.items():
        # Sort by date
        tile_products.sort(key=lambda x: x['acquisition_date'])

        print(f"\nTile {tile_id}: {len(tile_products)} products")
        for p in tile_products:
            print(f"  - {p['acquisition_date']}: {p['product_name']}")

        # Create pairs from consecutive acquisitions
        for i in range(len(tile_products) - 1):
            before = tile_products[i]
            after = tile_products[i + 1]

            # Calculate days apart
            before_date = before['acquisition_date']
            after_date = after['acquisition_date']

            from datetime import datetime
            date_before = datetime.strptime(before_date, "%Y%m%d")
            date_after = datetime.strptime(after_date, "%Y%m%d")
            days_apart = (date_after - date_before).days

            if days_apart <= max_days_apart:
                pair = {
                    'pair_id': pair_id,
                    'tile_id': tile_id,
                    'before': before,
                    'after': after,
                    'days_apart': days_apart,
                }
                pairs.append(pair)
                pair_id += 1

                print(f"\n  ✅ Pair {pair_id}: {before_date} → {after_date} ({days_apart} days)")

    return pairs


def tile_image(
    image: np.ndarray,
    patch_size: int = 512,
    overlap: int = 128
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Tile large image into smaller patches with overlap.

    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch (default: 512)
        overlap: Overlap between patches (default: 128)

    Returns:
        Tuple of (patches, positions)
        - patches: List of (patch_size, patch_size, C) arrays
        - positions: List of (y, x) top-left coordinates
    """
    H, W = image.shape[:2]
    stride = patch_size - overlap

    patches = []
    positions = []

    y_positions = list(range(0, H - patch_size + 1, stride))
    x_positions = list(range(0, W - patch_size + 1, stride))

    # Ensure we get the right edge
    if y_positions[-1] + patch_size < H:
        y_positions.append(H - patch_size)
    if x_positions[-1] + patch_size < W:
        x_positions.append(W - patch_size)

    for y in y_positions:
        for x in x_positions:
            patch = image[y:y+patch_size, x:x+patch_size]

            # Ensure patch is exactly patch_size × patch_size
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
                positions.append((y, x))

    return patches, positions


def process_pair(
    pair: Dict,
    output_dir: Path,
    patch_size: int = 512,
    overlap: int = 128
) -> Dict:
    """
    Process a single temporal pair:
        1. Load before/after images
        2. Tile into patches
        3. Save patches to A/ and B/ folders

    Args:
        pair: Pair metadata dictionary
        output_dir: Output directory (should contain A/ and B/ subdirs)
        patch_size: Patch size in pixels
        overlap: Overlap between patches

    Returns:
        Processing results dictionary
    """
    pair_id = pair['pair_id']
    tile_id = pair['tile_id']

    print(f"\n{'='*80}")
    print(f"Processing Pair {pair_id}: {tile_id}")
    print(f"  Before: {pair['before']['acquisition_date']}")
    print(f"  After:  {pair['after']['acquisition_date']}")
    print(f"  Days apart: {pair['days_apart']}")
    print(f"{'='*80}")

    # Load images
    before_path = Path(pair['before']['rgb_composite'])
    after_path = Path(pair['after']['rgb_composite'])

    print("Loading images...")
    img_before = np.array(Image.open(before_path))
    img_after = np.array(Image.open(after_path))

    print(f"  Before size: {img_before.shape}")
    print(f"  After size: {img_after.shape}")

    # Verify dimensions match
    if img_before.shape != img_after.shape:
        print(f"⚠️  Warning: Image dimensions don't match!")
        print(f"    Resizing after image to match before image...")
        img_after_pil = Image.fromarray(img_after)
        img_after_pil = img_after_pil.resize(
            (img_before.shape[1], img_before.shape[0]),
            Image.Resampling.LANCZOS
        )
        img_after = np.array(img_after_pil)

    # Tile images
    print(f"Tiling images (patch_size={patch_size}, overlap={overlap})...")
    patches_before, positions = tile_image(img_before, patch_size, overlap)
    patches_after, _ = tile_image(img_after, patch_size, overlap)

    print(f"  Created {len(patches_before)} patches")

    # Save patches
    a_dir = output_dir / "A"
    b_dir = output_dir / "B"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    patch_ids = []

    print("Saving patches...")
    for idx, (patch_a, patch_b) in enumerate(tqdm(
        zip(patches_before, patches_after),
        total=len(patches_before),
        desc="Saving"
    )):
        # Generate patch filename: pair{ID}_patch{INDEX}
        patch_id = f"pair{pair_id:02d}_patch{idx:04d}"
        patch_ids.append(patch_id)

        # Save to PNG
        Image.fromarray(patch_a).save(a_dir / f"{patch_id}.png", compress_level=6)
        Image.fromarray(patch_b).save(b_dir / f"{patch_id}.png", compress_level=6)

    print(f"✅ Saved {len(patch_ids)} patch pairs")

    return {
        'pair_id': pair_id,
        'tile_id': tile_id,
        'num_patches': len(patch_ids),
        'patch_ids': patch_ids,
        'patch_size': patch_size,
        'overlap': overlap,
        'before_date': pair['before']['acquisition_date'],
        'after_date': pair['after']['acquisition_date'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create temporal pairs and tile into patches for RCDNet'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='showcase/data/raw',
        help='Directory containing extracted RGB composites'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='showcase/data',
        help='Output directory (will create A/ and B/ subdirs)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=512,
        help='Patch size in pixels (default: 512)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=128,
        help='Overlap between patches (default: 128)'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=30,
        help='Maximum days between before/after images (default: 30)'
    )

    args = parser.parse_args()

    raw_dir = Path(args.input)
    output_dir = Path(args.output)

    if not raw_dir.exists():
        print(f"❌ Input directory not found: {raw_dir}")
        sys.exit(1)

    # Load product metadata
    print("Loading product metadata...")
    products = load_product_metadata(raw_dir)

    if not products:
        print(f"❌ No products found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(products)} products")

    # Find temporal pairs
    print("\nFinding temporal pairs...")
    pairs = find_temporal_pairs(products, max_days_apart=args.max_days)

    if not pairs:
        print("❌ No valid temporal pairs found")
        sys.exit(1)

    print(f"\n✅ Found {len(pairs)} temporal pairs")

    # Process all pairs
    results = []
    all_patch_ids = []

    for pair in pairs:
        result = process_pair(
            pair,
            output_dir,
            patch_size=args.patch_size,
            overlap=args.overlap
        )
        results.append(result)
        all_patch_ids.extend(result['patch_ids'])

    # Save pairs.txt (list of patch IDs)
    pairs_file = output_dir / "pairs.txt"
    print(f"\nSaving patch list to {pairs_file}...")
    with open(pairs_file, 'w') as f:
        for patch_id in all_patch_ids:
            f.write(f"{patch_id}\n")

    # Save metadata
    metadata = {
        'num_pairs': len(pairs),
        'num_patches': len(all_patch_ids),
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'pairs': results,
    }

    metadata_file = output_dir / "pairing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("PAIRING SUMMARY")
    print(f"{'='*80}")
    print(f"Temporal pairs: {len(pairs)}")
    print(f"Total patches: {len(all_patch_ids)}")
    print(f"Patch size: {args.patch_size}×{args.patch_size}")
    print(f"Overlap: {args.overlap} pixels")
    print(f"\nOutput structure:")
    print(f"  A/ - Before images ({len(all_patch_ids)} patches)")
    print(f"  B/ - After images ({len(all_patch_ids)} patches)")
    print(f"  pairs.txt - Patch ID list")
    print(f"  pairing_metadata.json - Processing metadata")
    print(f"\n✅ Ready for RCDNet inference!")


if __name__ == "__main__":
    main()
