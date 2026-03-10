#!/usr/bin/env python3
"""
Test RCDNet inference on a single image pair
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.RCDNet import RCDNet_model
from safetensors.torch import load_file


def load_model(checkpoint_path, device='cuda'):
    """Load RCDNet model from safetensors checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = RCDNet_model(
        input_nc=3,
        output_nc=1,
        token_len=4,
        resnet_stages_num=4,
        with_pos='learned',
        enc_depth=1,
        dec_depth=8
    )

    # Load weights
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded on {device}")
    return model


def preprocess_image(img_path):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0

    # Transpose to CHW format
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = torch.from_numpy(img).unsqueeze(0)

    return img


def run_inference(model, img_a, img_b, device='cuda'):
    """Run change detection inference"""
    img_a = img_a.to(device)
    img_b = img_b.to(device)

    with torch.no_grad():
        output = model(img_a, img_b)

    # Get change map (binary)
    change_map = torch.sigmoid(output)
    change_map = (change_map > 0.5).float()

    return change_map


def main():
    """Test inference on first patch"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Paths
    checkpoint = Path('weights/SECOND-model.safetensors')
    img_a_path = Path('showcase/data/A/pair00_patch0000.png')
    img_b_path = Path('showcase/data/B/pair00_patch0000.png')
    output_path = Path('showcase/test_output.png')

    if not checkpoint.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint}")
        return 1

    if not img_a_path.exists() or not img_b_path.exists():
        print(f"❌ Error: Image pairs not found")
        return 1

    # Load model
    model = load_model(checkpoint, device)

    # Load images
    print(f"\nLoading images...")
    print(f"  Before: {img_a_path}")
    print(f"  After:  {img_b_path}")

    img_a = preprocess_image(img_a_path)
    img_b = preprocess_image(img_b_path)

    print(f"  Image shape: {img_a.shape}")

    # Run inference
    print(f"\nRunning inference...")
    change_map = run_inference(model, img_a, img_b, device)

    # Save output
    change_map_np = change_map.squeeze().cpu().numpy()
    change_map_np = (change_map_np * 255).astype(np.uint8)

    Image.fromarray(change_map_np).save(output_path)

    # Statistics
    total_pixels = change_map_np.size
    changed_pixels = np.sum(change_map_np > 0)
    change_percentage = (changed_pixels / total_pixels) * 100

    print(f"\n✅ Inference complete!")
    print(f"  Output saved: {output_path}")
    print(f"  Image size: {change_map_np.shape}")
    print(f"  Changed pixels: {changed_pixels:,} / {total_pixels:,} ({change_percentage:.2f}%)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
