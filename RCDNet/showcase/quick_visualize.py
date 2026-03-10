#!/usr/bin/env python3
"""
Quick visualization: Show before/after image pairs side-by-side
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def create_comparison(data_dir: Path, num_samples: int = 5):
    """Create side-by-side comparison of before/after images."""

    a_dir = data_dir / "A"
    b_dir = data_dir / "B"

    # Get first N patches
    patches = sorted(a_dir.glob("*.png"))[:num_samples]

    if not patches:
        print(f"No patches found in {a_dir}")
        return

    print(f"Creating visualizations for {len(patches)} patches...")

    # Create figure
    fig, axes = plt.subplots(len(patches), 2, figsize=(12, 6*len(patches)))
    if len(patches) == 1:
        axes = axes.reshape(1, -1)

    for idx, patch_path in enumerate(patches):
        patch_id = patch_path.stem

        # Load images
        img_before = Image.open(a_dir / f"{patch_id}.png")
        img_after = Image.open(b_dir / f"{patch_id}.png")

        # Plot
        axes[idx, 0].imshow(img_before)
        axes[idx, 0].set_title(f"BEFORE: 2023-06-24\n{patch_id}", fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(img_after)
        axes[idx, 1].set_title(f"AFTER: 2024-06-23\n{patch_id}", fontsize=10)
        axes[idx, 1].axis('off')

    plt.suptitle("Year-Over-Year Comparison: Île-de-France (365 days)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = Path("showcase/comparison_grid.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")

    # Also save individual comparisons
    output_dir = Path("showcase/individual_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch_path in enumerate(patches):
        patch_id = patch_path.stem

        img_before = Image.open(a_dir / f"{patch_id}.png")
        img_after = Image.open(b_dir / f"{patch_id}.png")

        # Create side-by-side
        w, h = img_before.size
        combined = Image.new('RGB', (w*2 + 20, h + 60), 'white')

        # Add images
        combined.paste(img_before, (0, 60))
        combined.paste(img_after, (w + 20, 60))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((w//2 - 100, 20), "2023-06-24", fill='black', font=font)
        draw.text((w + w//2 - 80, 20), "2024-06-23", fill='black', font=font)

        combined.save(output_dir / f"{patch_id}_comparison.png")

    print(f"✅ Saved {len(patches)} individual comparisons to: {output_dir}")
    print(f"\nTo view: open showcase/comparison_grid.png")


if __name__ == "__main__":
    data_dir = Path("showcase/data_comparison")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    create_comparison(data_dir, num_samples)
