#!/usr/bin/env python3
"""
Download Sentinel-2 data for Île-de-France region (2025)

Uses the Copernicus Data Space Ecosystem (CDSE) API.
Note: The old Copernicus Open Access Hub (sentinelsat original target) has been deprecated.

Requirements:
    pip install requests tqdm

Setup:
    1. Register at https://dataspace.copernicus.eu/
    2. Set your credentials as environment variables:
       export CDSE_USERNAME="your_email"
       export CDSE_PASSWORD="your_password"

Usage:
    python download_sentinel_idf.py              # Interactive mode
    python download_sentinel_idf.py --auto       # Auto-download without confirmation
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Île-de-France bounding box (approximate)
# Format: (min_lon, min_lat, max_lon, max_lat)
IDF_BBOX = (1.4462, 48.1201, 3.5591, 49.2414)

# Île-de-France polygon (WKT format for more precise queries)
IDF_POLYGON = """POLYGON((
    1.4462 48.1201,
    3.5591 48.1201,
    3.5591 49.2414,
    1.4462 49.2414,
    1.4462 48.1201
))""".replace("\n", "").replace("  ", "")


class CopernicusDataSpaceClient:
    """Client for Copernicus Data Space Ecosystem API."""

    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token: Optional[str] = None
        self.session = requests.Session()

    def authenticate(self) -> str:
        """Get access token from CDSE authentication server."""
        auth_data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }

        response = requests.post(
            self.AUTH_URL,
            data=auth_data,
            verify=True,
            allow_redirects=False
        )

        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}"
            })
            print("Authentication successful")
            return self.access_token
        else:
            raise Exception(
                f"Authentication failed (status {response.status_code}): {response.text}"
            )

    def search_sentinel2(
        self,
        polygon_wkt: str,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 30.0,
        product_type: str = "S2MSI2A",  # Level-2A (atmospherically corrected)
        max_results: int = 100
    ) -> list:
        """
        Search for Sentinel-2 products.

        Args:
            polygon_wkt: Area of interest in WKT format (POLYGON)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            product_type: S2MSI1C (Level-1C) or S2MSI2A (Level-2A)
            max_results: Maximum number of results to return

        Returns:
            List of product dictionaries
        """
        # Build OData filter
        filters = [
            "Collection/Name eq 'SENTINEL-2'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')",
            f"ContentDate/Start gt {start_date}T00:00:00.000Z",
            f"ContentDate/Start lt {end_date}T23:59:59.999Z",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud_cover})",
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
        ]

        filter_str = " and ".join(filters)

        params = {
            "$filter": filter_str,
            "$top": max_results,
            "$orderby": "ContentDate/Start asc"
        }

        print(f"Searching for Sentinel-2 products...")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Max cloud cover: {max_cloud_cover}%")
        print(f"  Product type: {product_type}")

        response = requests.get(self.CATALOGUE_URL, params=params)

        if response.status_code != 200:
            raise Exception(
                f"Search failed (status {response.status_code}): {response.text}"
            )

        products = response.json().get("value", [])
        print(f"Found {len(products)} products")

        return products

    def download_product(
        self,
        product_id: str,
        product_name: str,
        output_dir: Path,
        use_zip: bool = True
    ) -> Path:
        """
        Download a single product.

        Args:
            product_id: Product UUID
            product_name: Product name (for filename)
            output_dir: Directory to save the product
            use_zip: If True, download as zip; otherwise as raw value

        Returns:
            Path to downloaded file
        """
        if not self.access_token:
            self.authenticate()

        endpoint = "$zip" if use_zip else "$value"
        url = f"{self.DOWNLOAD_URL}({product_id})/{endpoint}"

        output_path = output_dir / f"{product_name}.zip"

        if output_path.exists():
            print(f"  Already exists: {output_path.name}")
            return output_path

        response = self.session.get(url, stream=True, allow_redirects=True)

        # If $zip fails with 404, try $value endpoint
        if response.status_code == 404 and use_zip:
            print(f"  Compressed version not available, trying raw download...")
            return self.download_product(product_id, product_name, output_dir, use_zip=False)

        if response.status_code != 200:
            raise Exception(
                f"Download failed (status {response.status_code}): {response.text}"
            )

        # Get file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"  {product_name[:50]}...",
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  Downloaded: {output_path.name}")
        return output_path

    def download_all(
        self,
        products: list,
        output_dir: Path,
        max_downloads: Optional[int] = None
    ) -> list:
        """
        Download multiple products.

        Args:
            products: List of product dictionaries from search
            output_dir: Directory to save products
            max_downloads: Maximum number to download (None for all)

        Returns:
            List of paths to downloaded files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        to_download = products[:max_downloads] if max_downloads else products
        downloaded = []

        print(f"\nDownloading {len(to_download)} products to {output_dir}")

        for i, product in enumerate(to_download, 1):
            print(f"\n[{i}/{len(to_download)}] {product['Name']}")
            try:
                path = self.download_product(
                    product_id=product["Id"],
                    product_name=product["Name"],
                    output_dir=output_dir
                )
                downloaded.append(path)
            except Exception as e:
                print(f"  Error: {e}")
                continue

        return downloaded


def load_credentials(credentials_file: str = "credentials.txt") -> tuple:
    """Load credentials from file or environment variables."""
    # Try loading from file first
    creds_path = Path(credentials_file)
    if creds_path.exists():
        username = None
        password = None
        with open(creds_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("CDSE_USERNAME="):
                    username = line.split("=", 1)[1].strip('"').strip("'")
                elif line.startswith("CDSE_PASSWORD="):
                    password = line.split("=", 1)[1].strip('"').strip("'")
        if username and password:
            return username, password

    # Fallback to environment variables
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")

    return username, password


def main():
    # Get credentials from credentials.txt or environment variables
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

    # Search parameters for Île-de-France
    # Note: Very recent products may not be available for download yet
    # Using 2024 data as 2025 products may still be processing
    search_params = {
        "polygon_wkt": IDF_POLYGON,
        "start_date": "2024-06-01",  # Summer months for better quality
        "end_date": "2024-08-31",
        "max_cloud_cover": 20.0,  # 20% max cloud cover
        "product_type": "S2MSI2A",  # Level-2A (atmospherically corrected)
        "max_results": 5  # Limit for testing
    }

    # Search for products
    products = client.search_sentinel2(**search_params)

    if not products:
        print("\nNo products found matching the criteria.")
        return

    # Display found products
    print("\n" + "=" * 80)
    print("Found products:")
    print("=" * 80)
    for i, p in enumerate(products[:10], 1):  # Show first 10
        print(f"{i}. {p['Name']}")
        print(f"   Date: {p.get('ContentDate', {}).get('Start', 'N/A')[:10]}")
        print(f"   ID: {p['Id']}")

    if len(products) > 10:
        print(f"   ... and {len(products) - 10} more")

    # Save product list to JSON for reference
    output_dir = Path("./sentinel_data_idf_2025")
    output_dir.mkdir(parents=True, exist_ok=True)

    products_file = output_dir / "products_list.json"
    with open(products_file, "w") as f:
        json.dump(products, f, indent=2)
    print(f"\nProduct list saved to: {products_file}")

    # Ask user before downloading (unless --auto flag is used)
    print(f"\nReady to download {len(products)} products.")
    print(f"Output directory: {output_dir}")

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
        products=products,
        output_dir=output_dir,
        max_downloads=None  # Download all, or set a limit
    )

    print(f"\n{'=' * 80}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded)} / {len(products)} products")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
