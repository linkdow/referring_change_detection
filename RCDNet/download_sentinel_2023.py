#!/usr/bin/env python3
"""
Download Sentinel-2 data for Île-de-France region (June 2023)
For year-over-year change detection comparison with 2024 data.

Uses the same tile (T31UEQ) as the 2024 data for consistency.
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Import the client from the main download script
sys.path.insert(0, str(Path(__file__).parent))
from download_sentinel_idf import (
    CopernicusDataSpaceClient,
    IDF_POLYGON,
    load_credentials
)


def main():
    print("=" * 80)
    print("Sentinel-2 Download: June 2023 (Île-de-France)")
    print("For year-over-year change detection (2023 vs 2024)")
    print("=" * 80)
    print()

    # Get credentials
    username, password = load_credentials()

    if not username or not password:
        print("Error: Credentials not found")
        print("\nOption 1: Create credentials.txt with:")
        print('  CDSE_USERNAME="your_email@example.com"')
        print('  CDSE_PASSWORD="your_password"')
        print("\nOption 2: Set environment variables:")
        print('  export CDSE_USERNAME="your_email@example.com"')
        print('  export CDSE_PASSWORD="your_password"')
        print("\nTo register, visit: https://dataspace.copernicus.eu/")
        return

    # Initialize client
    client = CopernicusDataSpaceClient(username, password)

    # Authenticate
    client.authenticate()

    # Search for June 2023 data (same period as 2024 for comparison)
    # Using the same T31UEQ tile that has 2024 data
    search_params = {
        "polygon_wkt": IDF_POLYGON,
        "start_date": "2023-06-15",  # Mid-June 2023
        "end_date": "2023-06-30",    # End of June 2023
        "max_cloud_cover": 20.0,
        "product_type": "S2MSI2A",
        "max_results": 10
    }

    print("\nSearching for products...")
    print(f"  Period: June 2023")
    print(f"  Purpose: Year-over-year change detection baseline")
    print()

    # Search for products
    products = client.search_sentinel2(**search_params)

    if not products:
        print("\nNo products found matching the criteria.")
        print("Try expanding the date range or increasing cloud cover threshold.")
        return

    # Filter for T31UEQ tile (the one we have 2024 data for)
    t31ueq_products = [p for p in products if 'T31UEQ' in p['Name']]

    if not t31ueq_products:
        print("\nNo products found for tile T31UEQ.")
        print("Showing all available products instead:")
        t31ueq_products = products

    # Display found products
    print("\n" + "=" * 80)
    print(f"Found {len(t31ueq_products)} products for change detection:")
    print("=" * 80)
    for i, p in enumerate(t31ueq_products, 1):
        name = p['Name']
        date = p.get('ContentDate', {}).get('Start', 'N/A')[:10]

        # Extract tile ID from name
        tile = "Unknown"
        if '_T' in name:
            tile = name.split('_T')[1][:6]

        print(f"{i}. {name[:60]}")
        print(f"   Date: {date}")
        print(f"   Tile: {tile}")
        print(f"   ID: {p['Id']}")
        print()

    # Save product list
    output_dir = Path("./sentinel_data_2023")
    output_dir.mkdir(parents=True, exist_ok=True)

    products_file = output_dir / "products_list.json"
    with open(products_file, "w") as f:
        json.dump(t31ueq_products, f, indent=2)
    print(f"Product list saved to: {products_file}")

    # Ask user before downloading
    print(f"\nReady to download {len(t31ueq_products)} products.")
    print(f"Output directory: {output_dir}")
    print()
    print("This will provide a 2023 baseline for comparing with 2024 data.")

    auto_download = "--auto" in sys.argv or "-y" in sys.argv

    if not auto_download:
        try:
            response = input("\nProceed with download? [y/N]: ").strip().lower()
            if response != "y":
                print("Download cancelled.")
                return
        except EOFError:
            print("\nNon-interactive mode detected. Use --auto flag to proceed automatically.")
            return
    else:
        print("\nAuto-download mode enabled. Starting download...")

    # Download products
    downloaded = client.download_all(
        products=t31ueq_products,
        output_dir=output_dir,
        max_downloads=None
    )

    print(f"\n{'=' * 80}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded)} / {len(t31ueq_products)} products")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run showcase/scripts/01_extract_sentinel.py on 2023 data")
    print("  2. Run showcase/scripts/02_create_pairs.py to create 2023→2024 pairs")
    print("  3. Run inference to detect year-over-year changes!")


if __name__ == "__main__":
    main()
