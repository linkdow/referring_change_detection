#!/usr/bin/env python3
"""
RCDNet Inference Script for Sentinel-2 Showcase
Runs change detection on temporal image pairs and saves results.

Usage:
    python 03_run_inference.py \
        --config configs.config_sentinel_showcase \
        --checkpoint weights/SECOND-model.safetensors \
        --output showcase/results
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataloader.dataloader import get_val_loader
from models.builder import EncoderDecoder as SegModel

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_model_state_dict(path: str):
    """Load state dict from .pt/.pth or .safetensors file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def load_config(config_path: str):
    """Load config module."""
    import importlib
    mod = importlib.import_module(config_path)
    if not hasattr(mod, "config"):
        raise ValueError(f"{config_path} must expose `config`")
    return mod.config


def create_output_dirs(output_dir: Path, class_names: List[str]):
    """Create output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create per-class directories
    change_maps_dir = output_dir / "change_maps"
    for class_name in class_names[1:]:  # Skip "Non-change"
        class_dir = change_maps_dir / class_name.lower().replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

    # Create other output directories
    (output_dir / "composite").mkdir(exist_ok=True)
    (output_dir / "statistics").mkdir(exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)

    return output_dir


def run_inference_on_pair(
    model: nn.Module,
    llm_model: nn.Module,
    tokenizer,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    class_names: List[str],
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Run inference on a single image pair for all classes.

    Args:
        model: RCDNet model
        llm_model: CLIP text encoder
        tokenizer: CLIP tokenizer
        img_a: Before image [1, C, H, W]
        img_b: After image [1, C, H, W]
        class_names: List of class names
        device: torch device
        threshold: Binary threshold

    Returns:
        Dictionary mapping class names to binary change masks (H, W)
    """
    model.eval()
    llm_model.eval()

    H, W = img_a.shape[2:]
    results = {}

    with torch.no_grad():
        # Process each class (skip "Non-change" at index 0)
        for class_idx, class_name in enumerate(class_names[1:], start=1):
            # Encode text caption
            caption = class_name.lower()
            inputs = tokenizer(
                [caption],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            caption_embedding = llm_model(**inputs).last_hidden_state

            # Run model
            output = model(img_a, img_b, caption_embedding)

            # Process output
            if isinstance(output, dict):
                logits = output.get("seg_logits", output.get("out"))
            else:
                logits = output

            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # [H, W]
            binary_mask = (probs > threshold).cpu().numpy().astype(np.uint8)

            results[class_name] = binary_mask

    return results


def save_change_masks(
    results: Dict[str, np.ndarray],
    patch_id: str,
    output_dir: Path
):
    """Save per-class change masks."""
    change_maps_dir = output_dir / "change_maps"

    for class_name, mask in results.items():
        class_dir = change_maps_dir / class_name.lower().replace(" ", "_")

        # Save binary mask (0=no change, 255=change)
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img, mode='L').save(
            class_dir / f"{patch_id}.png"
        )


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Colorize a class index mask with distinct colors.

    Args:
        mask: (H, W) array of class indices
        num_classes: Number of classes

    Returns:
        (H, W, 3) RGB image
    """
    # Define distinct colors for each class
    colors = [
        [0, 0, 0],         # 0: Background (black)
        [255, 0, 0],       # 1: Low Vegetation (red)
        [255, 165, 0],     # 2: Non-vegetated Ground (orange)
        [0, 255, 0],       # 3: Tree (green)
        [0, 0, 255],       # 4: Water (blue)
        [255, 255, 0],     # 5: Building (yellow)
        [255, 0, 255],     # 6: Playground (magenta)
    ]

    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for class_idx in range(min(num_classes, len(colors))):
        rgb[mask == class_idx] = colors[class_idx]

    return rgb


def create_composite_mask(
    results: Dict[str, np.ndarray],
    class_names: List[str]
) -> np.ndarray:
    """
    Create composite change mask with different colors for each class.

    Args:
        results: Dictionary of per-class binary masks
        class_names: List of class names

    Returns:
        RGB composite mask (H, W, 3)
    """
    H, W = list(results.values())[0].shape
    composite = np.zeros((H, W), dtype=np.uint8)

    # Assign class indices to changed pixels
    for class_idx, class_name in enumerate(class_names[1:], start=1):
        if class_name in results:
            mask = results[class_name]
            composite[mask > 0] = class_idx

    # Colorize
    composite_rgb = colorize_mask(composite, len(class_names))

    return composite_rgb


def compute_statistics(
    results: Dict[str, np.ndarray],
    patch_id: str,
    pixel_area_m2: float = 100.0  # 10m resolution
) -> Dict:
    """
    Compute change statistics for a patch.

    Args:
        results: Dictionary of per-class binary masks
        patch_id: Patch identifier
        pixel_area_m2: Area per pixel in square meters

    Returns:
        Statistics dictionary
    """
    total_pixels = list(results.values())[0].size
    total_changed = sum(mask.sum() for mask in results.values())

    per_class_stats = {}
    for class_name, mask in results.items():
        changed_pixels = int(mask.sum())
        changed_area_m2 = changed_pixels * pixel_area_m2
        changed_area_km2 = changed_area_m2 / 1e6

        per_class_stats[class_name] = {
            "changed_pixels": changed_pixels,
            "changed_area_m2": changed_area_m2,
            "changed_area_km2": changed_area_km2,
            "percentage": (changed_pixels / total_pixels) * 100
        }

    stats = {
        "patch_id": patch_id,
        "total_pixels": int(total_pixels),
        "total_changed_pixels": int(total_changed),
        "total_changed_percentage": (total_changed / total_pixels) * 100,
        "pixel_area_m2": pixel_area_m2,
        "per_class": per_class_stats
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run RCDNet inference on Sentinel-2 showcase data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs.config_sentinel_showcase",
        help="Config module path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/SECOND-model.safetensors",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="showcase/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for change detection"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch size (for single image inference, use 1)"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Process only first N patches (for testing)"
    )

    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    cfg = load_config(args.config)

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d).to(device)
    state_dict = load_model_state_dict(args.checkpoint)

    # Handle 'model' key if checkpoint was saved with optimizer
    if "model" in state_dict:
        state_dict = state_dict["model"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")

    model.eval()
    print("✅ Model loaded successfully")

    # Load CLIP text encoder
    print("\nLoading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    llm_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    llm_model.eval()
    print("✅ CLIP encoder loaded")

    # Load dataset
    print(f"\nLoading dataset from {cfg.root_folder}...")
    dataset = get_val_loader(cfg)
    print(f"✅ Loaded {len(dataset)} image pairs")

    # Create output directories
    output_dir = Path(args.output)
    create_output_dirs(output_dir, cfg.class_names)
    print(f"\n✅ Output directory: {output_dir}")

    # Process patches
    num_patches = min(len(dataset), args.max_patches) if args.max_patches else len(dataset)
    print(f"\n{'='*80}")
    print(f"Processing {num_patches} patches...")
    print(f"{'='*80}\n")

    all_statistics = []

    for idx in tqdm(range(num_patches), desc="Running inference"):
        # Load data
        data = dataset[idx]
        img_a = data["A"].unsqueeze(0).to(device)
        img_b = data["B"].unsqueeze(0).to(device)

        # Get patch ID from filename or index
        try:
            patch_id = data.get("id", f"patch_{idx:04d}")
        except:
            patch_id = f"patch_{idx:04d}"

        # Run inference
        results = run_inference_on_pair(
            model=model,
            llm_model=llm_model,
            tokenizer=tokenizer,
            img_a=img_a,
            img_b=img_b,
            class_names=cfg.class_names,
            device=device,
            threshold=args.threshold
        )

        # Save per-class masks
        save_change_masks(results, patch_id, output_dir)

        # Create and save composite
        composite_rgb = create_composite_mask(results, cfg.class_names)
        Image.fromarray(composite_rgb).save(
            output_dir / "composite" / f"{patch_id}_composite.png"
        )

        # Compute statistics
        stats = compute_statistics(results, patch_id)
        all_statistics.append(stats)

    # Aggregate statistics
    print(f"\n{'='*80}")
    print("Computing aggregate statistics...")
    print(f"{'='*80}\n")

    total_changed_pixels = sum(s["total_changed_pixels"] for s in all_statistics)
    total_pixels = sum(s["total_pixels"] for s in all_statistics)

    # Per-class aggregation
    per_class_aggregate = {}
    for class_name in cfg.class_names[1:]:
        class_stats = [
            s["per_class"][class_name]
            for s in all_statistics
            if class_name in s["per_class"]
        ]

        total_class_pixels = sum(cs["changed_pixels"] for cs in class_stats)
        total_class_area_km2 = sum(cs["changed_area_km2"] for cs in class_stats)

        per_class_aggregate[class_name] = {
            "total_changed_pixels": total_class_pixels,
            "total_changed_area_km2": total_class_area_km2,
            "percentage_of_total": (total_class_pixels / total_pixels) * 100
        }

    aggregate_stats = {
        "num_patches": num_patches,
        "total_pixels": int(total_pixels),
        "total_changed_pixels": int(total_changed_pixels),
        "total_changed_percentage": (total_changed_pixels / total_pixels) * 100,
        "total_area_km2": (total_pixels * 100) / 1e6,  # 10m resolution
        "total_changed_area_km2": (total_changed_pixels * 100) / 1e6,
        "per_class": per_class_aggregate,
        "patch_statistics": all_statistics
    }

    # Save statistics
    stats_file = output_dir / "statistics" / "change_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Processed: {num_patches} patches")
    print(f"Total area: {aggregate_stats['total_area_km2']:.2f} km²")
    print(f"Changed area: {aggregate_stats['total_changed_area_km2']:.4f} km²")
    print(f"Change percentage: {aggregate_stats['total_changed_percentage']:.2f}%")

    print("\nPer-class changes:")
    for class_name, stats in per_class_aggregate.items():
        if stats['total_changed_pixels'] > 0:
            print(f"  {class_name:30s}: {stats['total_changed_area_km2']:.4f} km² "
                  f"({stats['percentage_of_total']:.2f}%)")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Change maps: {output_dir / 'change_maps'}")
    print(f"  - Composites: {output_dir / 'composite'}")
    print(f"  - Statistics: {stats_file}")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
