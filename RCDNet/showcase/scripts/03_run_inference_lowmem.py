#!/usr/bin/env python3
"""
Memory-Optimized RCDNet Inference Script for 4GB GPU
Uses mixed precision (FP16) and aggressive memory management
"""

import os
import sys
import json
import argparse
import warnings
import gc
from pathlib import Path
from typing import Dict, List

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
    """Aggressively clear CUDA cache"""
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


def run_inference_single_class(
    model: nn.Module,
    llm_model: nn.Module,
    tokenizer,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    class_name: str,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True
) -> np.ndarray:
    """
    Run inference for a single class with memory optimization.

    Args:
        model: RCDNet model
        llm_model: CLIP text encoder
        tokenizer: CLIP tokenizer
        img_a: Before image [1, C, H, W]
        img_b: After image [1, C, H, W]
        class_name: Class name to detect
        device: torch device
        threshold: Binary threshold
        use_amp: Use automatic mixed precision (FP16)

    Returns:
        Binary change mask (H, W)
    """
    model.eval()
    llm_model.eval()

    with torch.no_grad():
        # Encode text caption with AMP
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

        # Clear intermediate tensors
        del inputs
        clear_cuda_cache()

        # Run model with AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(img_a, img_b, captions=caption_embedding)
        else:
            output = model(img_a, img_b, captions=caption_embedding)

        # Clear caption embedding
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

        # Clear output tensors
        del output, logits, probs
        clear_cuda_cache()

    return binary_mask


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized RCDNet inference")
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
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp

    print("Loading configuration...")
    cfg = load_config(args.config)

    print(f"Using device: {device}")
    if use_amp:
        print("✅ Mixed precision (FP16) enabled for memory efficiency")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d).to(device)

    state_dict = load_model_state_dict(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Clear initial cache
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

    print(f"\nLoaded {len(patch_ids)} image pairs")

    # Output setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in cfg.class_names[1:]:  # Skip "Non-change"
        class_dir = output_dir / "change_maps" / class_name.lower().replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Output directory: {output_dir}")

    # Process patches
    print(f"\n{'='*80}")
    print(f"Processing {len(patch_ids)} patches...")
    print(f"{'='*80}\n")

    stats = {cls: 0 for cls in cfg.class_names[1:]}

    for idx, patch_id in enumerate(tqdm(patch_ids, desc="Running inference")):
        # Load images
        img_a_path = data_root / "A" / f"{patch_id}.png"
        img_b_path = data_root / "B" / f"{patch_id}.png"

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        # Convert to tensor and normalize
        img_a = np.array(img_a, dtype=np.float32) / 255.0
        img_b = np.array(img_b, dtype=np.float32) / 255.0

        # Apply normalization
        for c in range(3):
            img_a[:, :, c] = (img_a[:, :, c] - cfg.norm_mean[c]) / cfg.norm_std[c]
            img_b[:, :, c] = (img_b[:, :, c] - cfg.norm_mean[c]) / cfg.norm_std[c]

        # To tensor [1, C, H, W]
        img_a = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).to(device)
        img_b = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).to(device)

        # Process each class separately for memory efficiency
        for class_name in cfg.class_names[1:]:  # Skip "Non-change"
            mask = run_inference_single_class(
                model, llm_model, tokenizer,
                img_a, img_b, class_name,
                device, args.threshold, use_amp
            )

            # Save if any changes detected
            if mask.sum() > 0:
                class_dir = output_dir / "change_maps" / class_name.lower().replace(" ", "_")
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(class_dir / f"{patch_id}.png")
                stats[class_name] += 1

        # Clear image tensors
        del img_a, img_b
        clear_cuda_cache()

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
            "threshold": args.threshold,
            "device": str(device),
            "mixed_precision": use_amp,
            "detections_per_class": stats
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"✅ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
