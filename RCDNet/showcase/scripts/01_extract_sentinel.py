#!/usr/bin/env python3
"""
Sentinel-2 Band Extraction Script
Extracts RGB bands (B02, B03, B04) from Sentinel-2 SAFE products and creates RGB composites.

Usage:
    python 01_extract_sentinel.py --input sentinel_data_idf_2025 --output showcase/data/raw
"""

import os
import sys
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from PIL import Image


def find_band_files(safe_dir: Path, bands: list = ['B02', 'B03', 'B04']) -> Dict[str, Path]:
    """
    Find band files in SAFE directory structure.

    Args:
        safe_dir: Path to extracted SAFE directory
        bands: List of band names to find (default: RGB bands)

    Returns:
        Dictionary mapping band name to file path
    """
    band_files = {}

    # Search in GRANULE/*/IMG_DATA/R10m/ for 10m resolution bands
    img_data_pattern = safe_dir / "GRANULE" / "*" / "IMG_DATA" / "R10m"

    for granule_dir in safe_dir.glob("GRANULE/*"):
        r10m_dir = granule_dir / "IMG_DATA" / "R10m"

        if not r10m_dir.exists():
            continue

        for band in bands:
            # Pattern: T31UEQ_20240623T104619_{BAND}_10m.jp2
            band_pattern = f"*_{band}_10m.jp2"
            matches = list(r10m_dir.glob(band_pattern))

            if matches:
                band_files[band] = matches[0]
                print(f"  Found {band}: {matches[0].name}")

    return band_files


def read_band(band_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read a single Sentinel-2 band using rasterio.

    Args:
        band_path: Path to JP2 band file

    Returns:
        Tuple of (band_array, metadata)
    """
    with rasterio.open(band_path) as src:
        band_data = src.read(1)  # Read first (and only) band
        metadata = {
            'crs': str(src.crs),
            'transform': list(src.transform),
            'width': src.width,
            'height': src.height,
            'bounds': list(src.bounds),
        }

    return band_data, metadata


def create_rgb_composite(
    bands: Dict[str, np.ndarray],
    percentile_clip: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Create RGB composite from individual bands with contrast stretching.

    Args:
        bands: Dictionary with 'B04' (Red), 'B03' (Green), 'B02' (Blue) arrays
        percentile_clip: Percentile values for contrast stretching

    Returns:
        RGB composite array (H, W, 3) with values in [0, 255]
    """
    # Stack bands: R, G, B
    rgb = np.stack([
        bands['B04'],  # Red
        bands['B03'],  # Green
        bands['B02'],  # Blue
    ], axis=-1).astype(np.float32)

    # Contrast stretching per channel
    for i in range(3):
        channel = rgb[:, :, i]
        p_low, p_high = np.percentile(channel, percentile_clip)

        # Clip and normalize to [0, 255]
        channel = np.clip(channel, p_low, p_high)
        channel = ((channel - p_low) / (p_high - p_low) * 255)
        rgb[:, :, i] = channel

    return rgb.astype(np.uint8)


def extract_product_metadata(safe_dir: Path) -> Dict:
    """
    Extract metadata from Sentinel-2 product name and MTD file.

    Product name format:
    S2B_MSIL2A_20240623T104619_N0510_R051_T31UEQ_20240623T122156

    Args:
        safe_dir: Path to SAFE directory

    Returns:
        Metadata dictionary
    """
    product_name = safe_dir.name.replace('.SAFE', '')
    parts = product_name.split('_')

    metadata = {
        'product_name': product_name,
        'satellite': parts[0],  # S2A or S2B
        'processing_level': parts[1],  # MSIL2A
        'acquisition_date': parts[2][:8],  # YYYYMMDD
        'acquisition_time': parts[2][9:],  # HHMMSS
        'processing_baseline': parts[3],  # N0510
        'relative_orbit': parts[4],  # R051
        'tile_id': parts[5],  # T31UEQ
        'processing_date': parts[6][:8] if len(parts) > 6 else None,
    }

    return metadata


def process_sentinel_product(
    zip_path: Path,
    output_dir: Path,
    keep_temp: bool = False
) -> Dict:
    """
    Process a single Sentinel-2 product.

    Args:
        zip_path: Path to SAFE.zip file
        output_dir: Directory to save outputs
        keep_temp: Keep temporary extracted files

    Returns:
        Processing results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Processing: {zip_path.name}")
    print(f"{'='*80}")

    # Create temporary directory for extraction
    temp_dir = Path(tempfile.mkdtemp(prefix='sentinel_'))

    try:
        # Extract SAFE archive
        print("Extracting SAFE archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find SAFE directory
        safe_dirs = list(temp_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise ValueError("No .SAFE directory found in archive")

        safe_dir = safe_dirs[0]
        print(f"SAFE directory: {safe_dir.name}")

        # Extract metadata
        metadata = extract_product_metadata(safe_dir)
        print(f"Tile: {metadata['tile_id']}, Date: {metadata['acquisition_date']}")

        # Find band files
        print("\nSearching for RGB bands...")
        band_files = find_band_files(safe_dir, bands=['B02', 'B03', 'B04'])

        if len(band_files) != 3:
            raise ValueError(f"Expected 3 bands, found {len(band_files)}: {list(band_files.keys())}")

        # Read bands
        print("\nReading bands...")
        bands = {}
        geo_metadata = None

        for band_name, band_path in band_files.items():
            print(f"  Reading {band_name}...")
            band_data, band_meta = read_band(band_path)
            bands[band_name] = band_data

            if geo_metadata is None:
                geo_metadata = band_meta

        print(f"  Image size: {geo_metadata['width']} × {geo_metadata['height']}")
        print(f"  CRS: {geo_metadata['crs']}")

        # Create RGB composite
        print("\nCreating RGB composite...")
        rgb_composite = create_rgb_composite(bands)

        # Generate output filename
        output_filename = f"{metadata['tile_id']}_{metadata['acquisition_date']}.png"
        output_path = output_dir / output_filename
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save RGB composite
        print(f"Saving to: {output_path}")
        Image.fromarray(rgb_composite).save(output_path, compress_level=6)

        # Save metadata
        metadata_full = {
            **metadata,
            'geo': geo_metadata,
            'rgb_composite': str(output_path),
            'original_zip': str(zip_path),
        }

        metadata_path = output_dir / f"{metadata['tile_id']}_{metadata['acquisition_date']}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_full, f, indent=2)

        print(f"✅ Success! RGB composite saved: {output_filename}")

        return {
            'status': 'success',
            'output_image': output_path,
            'output_metadata': metadata_path,
            'metadata': metadata_full,
        }

    except Exception as e:
        print(f"❌ Error processing {zip_path.name}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'zip_path': str(zip_path),
        }

    finally:
        # Clean up temporary directory
        if not keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Extract RGB bands from Sentinel-2 SAFE products'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='sentinel_data_idf_2025',
        help='Directory containing Sentinel-2 zip files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='showcase/data/raw',
        help='Output directory for RGB composites'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary extracted files'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Find all Sentinel-2 zip files
    zip_files = sorted(input_dir.glob("S2*.zip"))

    if not zip_files:
        print(f"❌ No Sentinel-2 zip files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(zip_files)} Sentinel-2 products")
    print(f"Output directory: {output_dir}")
    print()

    # Process all products
    results = []
    for zip_path in zip_files:
        result = process_sentinel_product(zip_path, output_dir, args.keep_temp)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count

    print(f"Total products: {len(results)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")

    if success_count > 0:
        print(f"\nRGB composites saved to: {output_dir}")

        # Save processing summary
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Summary saved to: {summary_path}")

    if error_count > 0:
        print("\n⚠️  Some products failed to process. Check the summary for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
