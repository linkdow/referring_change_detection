#!/usr/bin/env python3
"""
Tiled RCDNet Inference for 4GB GPU
Splits 512×512 images into 4×256×256 tiles, runs inference, and stitches results.
"""

import os
import sys
import json
import argparse
import warnings
import gc
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

from models.builder import EncoderDecoder as SegModel

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


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


def split_into_tiles(image: np.ndarray, tile_size: int = 256) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split a 512×512 image into 4 non-overlapping 256×256 tiles.

    Args:
        image: Input image [H, W, C]
        tile_size: Size of each tile (default 256)

    Returns:
        List of (tile, (row, col)) tuples
    """
    H, W = image.shape[:2]
    tiles = []

    # Calculate grid dimensions
    rows = H // tile_size
    cols = W // tile_size

    for i in range(rows):
        for j in range(cols):
            y_start = i * tile_size
            y_end = y_start + tile_size
            x_start = j * tile_size
            x_end = x_start + tile_size

            tile = image[y_start:y_end, x_start:x_end]
            tiles.append((tile, (i, j)))

    return tiles


def stitch_tiles(tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
                output_size: Tuple[int, int] = (512, 512),
                tile_size: int = 256) -> np.ndarray:
    """
    Stitch tiles back into a full image.

    Args:
        tiles: List of (tile, (row, col)) tuples
        output_size: Size of output image (H, W)
        tile_size: Size of each tile

    Returns:
        Stitched image [H, W]
    """
    H, W = output_size
    output = np.zeros((H, W), dtype=np.uint8)

    for tile, (i, j) in tiles:
        y_start = i * tile_size
        y_end = y_start + tile_size
        x_start = j * tile_size
        x_end = x_start + tile_size

        output[y_start:y_end, x_start:x_end] = tile

    return output


def run_inference_on_tile(
    model: nn.Module,
    llm_model: nn.Module,
    tokenizer,
    tile_a: torch.Tensor,
    tile_b: torch.Tensor,
    class_name: str,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True
) -> np.ndarray:
    """
    Run inference on a single 256×256 tile.

    Returns:
        Binary change mask (256, 256)
    """
    model.eval()
    llm_model.eval()

    with torch.no_grad():
        # Encode text caption
        caption = class_name.lower()
        inputs = tokenizer(
            [caption],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                caption_embedding = llm_model(**inputs).last_hidden_state
        else:
            caption_embedding = llm_model(**inputs).last_hidden_state

        del inputs
        clear_cuda_cache()

        # Run model (pass None for label to indicate inference mode)
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(tile_a, tile_b, label=None, captions=caption_embedding)
        else:
            output = model(tile_a, tile_b, label=None, captions=caption_embedding)

        del caption_embedding
        clear_cuda_cache()

        # Process output
        if isinstance(output, dict):
            logits = output.get("seg_logits", output.get("out"))
        else:
            logits = output

        # Apply sigmoid and threshold
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # [H, W]
        binary_mask = (probs > threshold).cpu().numpy().astype(np.uint8)

        del output, logits, probs
        clear_cuda_cache()

    return binary_mask


def process_patch_tiled(
    model: nn.Module,
    llm_model: nn.Module,
    tokenizer,
    img_a: np.ndarray,
    img_b: np.ndarray,
    class_names: List[str],
    cfg,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True,
    tile_size: int = 256
) -> Dict[str, np.ndarray]:
    """
    Process a 512×512 patch using tiled inference.

    Args:
        img_a, img_b: Input images [512, 512, 3]
        class_names: List of class names
        cfg: Config object
        device: Torch device
        threshold: Detection threshold
        use_amp: Use automatic mixed precision
        tile_size: Size of tiles (256)

    Returns:
        Dictionary mapping class names to 512×512 binary masks
    """
    results = {}

    # Split images into tiles
    tiles_a = split_into_tiles(img_a, tile_size)
    tiles_b = split_into_tiles(img_b, tile_size)

    assert len(tiles_a) == len(tiles_b), "Tile count mismatch"

    # Process each class
    for class_name in class_names[1:]:  # Skip "Non-change"
        class_tiles = []

        # Process each tile
        for (tile_a, pos), (tile_b, _) in zip(tiles_a, tiles_b):
            # Normalize tile
            tile_a_norm = tile_a.copy().astype(np.float32) / 255.0
            tile_b_norm = tile_b.copy().astype(np.float32) / 255.0

            for c in range(3):
                tile_a_norm[:, :, c] = (tile_a_norm[:, :, c] - cfg.norm_mean[c]) / cfg.norm_std[c]
                tile_b_norm[:, :, c] = (tile_b_norm[:, :, c] - cfg.norm_mean[c]) / cfg.norm_std[c]

            # Convert to tensor [1, C, H, W]
            tile_a_tensor = torch.from_numpy(tile_a_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            tile_b_tensor = torch.from_numpy(tile_b_norm).permute(2, 0, 1).unsqueeze(0).to(device)

            # Run inference on tile
            tile_mask = run_inference_on_tile(
                model, llm_model, tokenizer,
                tile_a_tensor, tile_b_tensor,
                class_name, device, threshold, use_amp
            )

            class_tiles.append((tile_mask, pos))

            del tile_a_tensor, tile_b_tensor, tile_a_norm, tile_b_norm
            clear_cuda_cache()

        # Stitch tiles back together
        full_mask = stitch_tiles(class_tiles, output_size=(512, 512), tile_size=tile_size)
        results[class_name] = full_mask

    return results


def main():
    parser = argparse.ArgumentParser(description="Tiled RCDNet inference for 4GB GPU")
    parser.add_argument("--config", type=str, default="configs.config_sentinel_showcase",
                        help="Config module path")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="showcase/results",
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Change detection threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Maximum patches to process (for testing)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Tile size (default: 256)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp

    print("Loading configuration...")
    cfg = load_config(args.config)

    print(f"Using device: {device}")
    print(f"Tile size: {args.tile_size}×{args.tile_size}")
    if use_amp:
        print("✅ Mixed precision (FP16) enabled")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d).to(device)

    state_dict = load_model_state_dict(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    clear_cuda_cache()
    print(f"✅ Model loaded successfully")

    # Load CLIP
    print("\nLoading CLIP text encoder...")
    llm_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    llm_model.eval()

    clear_cuda_cache()
    print("✅ CLIP encoder loaded")

    # Load dataset paths
    data_root = Path(cfg.root_folder)
    pairs_file = data_root / "pairs.txt"

    with open(pairs_file) as f:
        patch_ids = [line.strip() for line in f if line.strip()]

    if args.max_patches:
        patch_ids = patch_ids[:args.max_patches]

    print(f"\n✅ Loaded {len(patch_ids)} image pairs")

    # Output setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in cfg.class_names[1:]:  # Skip "Non-change"
        class_dir = output_dir / "change_maps" / class_name.lower().replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Output directory: {output_dir}")

    # Process patches
    print(f"\n{'='*80}")
    print(f"Processing {len(patch_ids)} patches (tiled inference)...")
    print(f"{'='*80}\n")

    stats = {cls: 0 for cls in cfg.class_names[1:]}

    for patch_id in tqdm(patch_ids, desc="Running tiled inference"):
        # Load images
        img_a_path = data_root / "A" / f"{patch_id}.png"
        img_b_path = data_root / "B" / f"{patch_id}.png"

        img_a = np.array(Image.open(img_a_path).convert("RGB"))
        img_b = np.array(Image.open(img_b_path).convert("RGB"))

        # Run tiled inference
        results = process_patch_tiled(
            model, llm_model, tokenizer,
            img_a, img_b,
            cfg.class_names, cfg,
            device, args.threshold, use_amp,
            args.tile_size
        )

        # Save results
        for class_name, mask in results.items():
            if mask.sum() > 0:  # Only save if changes detected
                class_dir = output_dir / "change_maps" / class_name.lower().replace(" ", "_")
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(class_dir / f"{patch_id}.png")
                stats[class_name] += 1

    # Save statistics
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}\n")
    print("Changes detected per class:")
    for class_name, count in stats.items():
        percentage = (count / len(patch_ids)) * 100
        print(f"  {class_name:30s}: {count:4d} patches ({percentage:5.1f}%)")

    stats_file = output_dir / "statistics" / "detection_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file, 'w') as f:
        json.dump({
            "total_patches": len(patch_ids),
            "tile_size": args.tile_size,
            "threshold": args.threshold,
            "device": str(device),
            "mixed_precision": use_amp,
            "detections_per_class": stats
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"✅ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
