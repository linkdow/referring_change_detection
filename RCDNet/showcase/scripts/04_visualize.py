#!/usr/bin/env python3
"""
Visualization Script for RCDNet Sentinel-2 Showcase
Creates publication-ready figures from inference results.

Usage:
    python 04_visualize.py \
        --data showcase/data \
        --results showcase/results \
        --output showcase/visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from tqdm import tqdm


def load_statistics(stats_file: Path) -> Dict:
    """Load change statistics from JSON file."""
    with open(stats_file) as f:
        return json.load(f)


def create_before_after_comparison(
    data_dir: Path,
    results_dir: Path,
    output_file: Path,
    patch_ids: List[str] = None,
    num_samples: int = 6
):
    """
    Create before/after comparison grid with change overlays.

    Args:
        data_dir: Directory containing A/ and B/ folders
        results_dir: Directory containing change_maps/
        output_file: Output file path
        patch_ids: List of patch IDs to visualize (None = random selection)
        num_samples: Number of samples to show
    """
    # Get available patches
    a_dir = data_dir / "A"
    all_patches = sorted([p.stem for p in a_dir.glob("*.png")])

    if patch_ids is None:
        # Select patches with most changes
        change_counts = []
        composite_dir = results_dir / "composite"

        for patch_id in all_patches[:100]:  # Check first 100 for speed
            composite_path = composite_dir / f"{patch_id}_composite.png"
            if composite_path.exists():
                composite = np.array(Image.open(composite_path))
                # Count non-black pixels (changes)
                change_pixels = np.sum(np.any(composite > 0, axis=-1))
                change_counts.append((patch_id, change_pixels))

        # Sort by change count and take top N
        change_counts.sort(key=lambda x: x[1], reverse=True)
        patch_ids = [pc[0] for pc in change_counts[:num_samples]]

    # Create figure
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 3, figure=fig, hspace=0.3, wspace=0.1)

    for idx, patch_id in enumerate(patch_ids):
        # Load images
        img_a = Image.open(data_dir / "A" / f"{patch_id}.png")
        img_b = Image.open(data_dir / "B" / f"{patch_id}.png")
        composite_path = results_dir / "composite" / f"{patch_id}_composite.png"

        if composite_path.exists():
            composite = Image.open(composite_path)
        else:
            composite = Image.new('RGB', img_b.size, color='black')

        # Plot before
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(img_a)
        ax1.set_title(f"Before (June 23)", fontsize=12)
        ax1.axis('off')

        # Plot after
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(img_b)
        ax2.set_title(f"After (June 25)", fontsize=12)
        ax2.axis('off')

        # Plot overlay
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(img_b)
        ax3.imshow(composite, alpha=0.5)
        ax3.set_title(f"Change Detection", fontsize=12)
        ax3.axis('off')

        # Add patch ID as ylabel
        ax1.set_ylabel(f"{patch_id}", fontsize=10, rotation=0,
                       ha='right', va='center', labelpad=20)

    plt.suptitle("IGN Orthophoto Change Detection: Saint-Denis / Village Olympique (2021 → 2024)",
                 fontsize=16, fontweight='bold')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Created comparison grid: {output_file}")


def create_class_breakdown_chart(
    stats: Dict,
    output_file: Path
):
    """
    Create bar chart showing changes per class.

    Args:
        stats: Statistics dictionary
        output_file: Output file path
    """
    per_class = stats['per_class']

    # Extract data
    classes = []
    areas_km2 = []
    percentages = []

    for class_name, class_stats in per_class.items():
        if class_stats['total_changed_pixels'] > 0:  # Only show classes with changes
            classes.append(class_name)
            areas_km2.append(class_stats['total_changed_area_km2'])
            percentages.append(class_stats['percentage_of_total'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Changed area (km²)
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars1 = ax1.barh(classes, areas_km2, color=colors)
    ax1.set_xlabel('Changed Area (km²)', fontsize=12)
    ax1.set_title('Total Changed Area by Class', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, area in zip(bars1, areas_km2):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{area:.4f} km²',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # Plot 2: Percentage of total area
    bars2 = ax2.barh(classes, percentages, color=colors)
    ax2.set_xlabel('Percentage of Total Area (%)', fontsize=12)
    ax2.set_title('Change Percentage by Class', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, pct in zip(bars2, percentages):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{pct:.2f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')

    plt.suptitle(f"Change Detection Statistics\n"
                 f"Total Area: {stats['total_area_km2']:.2f} km² | "
                 f"Changed: {stats['total_changed_area_km2']:.4f} km² "
                 f"({stats['total_changed_percentage']:.2f}%)",
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Created class breakdown chart: {output_file}")


def create_change_heatmap(
    results_dir: Path,
    output_file: Path,
    patch_size: int = 512,
    overlap: int = 128
):
    """
    Create aggregate change intensity heatmap from all patches.

    Args:
        results_dir: Directory containing composite images
        output_file: Output file path
        patch_size: Size of each patch
        overlap: Overlap between patches
    """
    composite_dir = results_dir / "composite"
    composite_files = sorted(composite_dir.glob("*_composite.png"))

    if not composite_files:
        print("⚠️  No composite images found, skipping heatmap")
        return

    # Load first image to get dimensions
    first_img = Image.open(composite_files[0])
    H, W = first_img.size

    # Calculate grid dimensions (rough estimate based on patch count)
    num_patches = len(composite_files)
    grid_size = int(np.ceil(np.sqrt(num_patches)))

    # Create heatmap
    heatmap = np.zeros((grid_size * H, grid_size * W), dtype=np.float32)

    print("Creating change intensity heatmap...")
    for idx, composite_file in enumerate(tqdm(composite_files, desc="Processing")):
        composite = np.array(Image.open(composite_file))

        # Calculate change intensity (any non-black pixel)
        intensity = np.any(composite > 0, axis=-1).astype(np.float32)

        # Place in grid
        row = idx // grid_size
        col = idx % grid_size

        y_start = row * H
        x_start = col * W

        heatmap[y_start:y_start+H, x_start:x_start+W] = intensity

    # Plot
    fig, ax = plt.subplots(figsize=(16, 16))

    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title('Change Intensity Heatmap\nÎle-de-France Region',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Change Detected', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Created change heatmap: {output_file}")


def create_per_class_maps(
    results_dir: Path,
    output_dir: Path,
    class_names: List[str],
    max_samples: int = 5
):
    """
    Create individual visualizations for each class.

    Args:
        results_dir: Directory containing change_maps/
        output_dir: Output directory for per-class visualizations
        class_names: List of class names
        max_samples: Number of sample patches per class
    """
    change_maps_dir = results_dir / "change_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_dir = change_maps_dir / class_name.lower().replace(" ", "_")

        if not class_dir.exists():
            continue

        class_files = sorted(class_dir.glob("*.png"))

        if not class_files:
            continue

        # Select samples with most changes
        change_counts = []
        for f in class_files[:50]:  # Check first 50
            mask = np.array(Image.open(f))
            change_counts.append((f, mask.sum()))

        change_counts.sort(key=lambda x: x[1], reverse=True)
        selected = change_counts[:max_samples]

        # Create visualization
        fig, axes = plt.subplots(1, max_samples, figsize=(4*max_samples, 4))

        if max_samples == 1:
            axes = [axes]

        for ax, (file_path, change_sum) in zip(axes, selected):
            mask = Image.open(file_path)
            ax.imshow(mask, cmap='Reds')
            ax.set_title(f"{file_path.stem}\n{change_sum/255:.0f} changed pixels",
                        fontsize=10)
            ax.axis('off')

        plt.suptitle(f"{class_name} Change Detection",
                    fontsize=14, fontweight='bold')

        output_file = output_dir / f"{class_name.lower().replace(' ', '_')}_samples.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Created {class_name} visualization: {output_file}")


def create_summary_report(
    stats: Dict,
    output_file: Path
):
    """
    Create text summary report.

    Args:
        stats: Statistics dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IGN ORTHOPHOTO CHANGE DETECTION SUMMARY\n")
        f.write("Saint-Denis / Village Olympique (2021 → 2024)\n")
        f.write("="*80 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of patches analyzed: {stats['num_patches']}\n")
        f.write(f"Total area covered: {stats['total_area_km2']:.2f} km²\n")
        f.write(f"Total changed area: {stats['total_changed_area_km2']:.4f} km²\n")
        f.write(f"Overall change percentage: {stats['total_changed_percentage']:.2f}%\n")
        f.write("\n")

        f.write("PER-CLASS BREAKDOWN\n")
        f.write("-"*80 + "\n")

        per_class = stats['per_class']
        for class_name, class_stats in sorted(per_class.items(),
                                              key=lambda x: x[1]['total_changed_area_km2'],
                                              reverse=True):
            if class_stats['total_changed_pixels'] > 0:
                f.write(f"\n{class_name}:\n")
                f.write(f"  Changed area: {class_stats['total_changed_area_km2']:.4f} km²\n")
                f.write(f"  Percentage: {class_stats['percentage_of_total']:.2f}%\n")
                f.write(f"  Changed pixels: {class_stats['total_changed_pixels']:,}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✅ Created summary report: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations from RCDNet inference results"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="showcase/data",
        help="Data directory containing A/ and B/ folders"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="showcase/results",
        help="Results directory from inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="showcase/visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of sample patches for comparison grid"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    results_dir = Path(args.results)
    output_dir = Path(args.output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Load statistics
    stats_file = results_dir / "statistics" / "change_statistics.json"
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        print("Please run inference first (03_run_inference.py)")
        sys.exit(1)

    stats = load_statistics(stats_file)

    # 1. Before/After comparison grid
    print("\n[1/5] Creating before/after comparison grid...")
    create_before_after_comparison(
        data_dir=data_dir,
        results_dir=results_dir,
        output_file=output_dir / "before_after_comparison.png",
        num_samples=args.num_samples
    )

    # 2. Class breakdown chart
    print("\n[2/5] Creating class breakdown chart...")
    create_class_breakdown_chart(
        stats=stats,
        output_file=output_dir / "class_breakdown.png"
    )

    # 3. Change heatmap
    print("\n[3/5] Creating change intensity heatmap...")
    create_change_heatmap(
        results_dir=results_dir,
        output_file=output_dir / "change_heatmap.png"
    )

    # 4. Per-class visualizations
    print("\n[4/5] Creating per-class visualizations...")
    class_names = [
        "Low Vegetation", "Non-vegetated Ground Surface",
        "Tree", "Water", "Building", "Playground"
    ]
    create_per_class_maps(
        results_dir=results_dir,
        output_dir=output_dir / "per_class",
        class_names=class_names,
        max_samples=5
    )

    # 5. Summary report
    print("\n[5/5] Creating summary report...")
    create_summary_report(
        stats=stats,
        output_file=output_dir / "summary_report.txt"
    )

    # Final summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - before_after_comparison.png    # Side-by-side comparison grid")
    print("  - class_breakdown.png            # Bar charts of changes per class")
    print("  - change_heatmap.png             # Aggregate change intensity map")
    print("  - per_class/                     # Individual class visualizations")
    print("  - summary_report.txt             # Text summary of results")
    print("\n✅ All visualizations created successfully!")


if __name__ == "__main__":
    main()
