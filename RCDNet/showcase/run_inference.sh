#!/bin/bash
# RCDNet Inference Runner for 4GB GPU
# Uses tiled inference to fit in limited GPU memory

export PYTHONPATH=/home/noam/referring_change_detection/RCDNet/models/encoders/selective_scan:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "Starting RCDNet inference on 841 Sentinel-2 patches..."
echo "Device: CUDA (4GB GPU)"
echo "Strategy: Tiled inference (512×512 → 4×256×256)"
echo ""

python3 showcase/scripts/03_run_inference_tiled.py \
    --config configs.config_sentinel_showcase \
    --checkpoint weights/SECOND-model.safetensors \
    --output showcase/results \
    --device cuda \
    --tile-size 256

echo ""
echo "✅ Inference complete!"
echo "Results saved to: showcase/results/"
