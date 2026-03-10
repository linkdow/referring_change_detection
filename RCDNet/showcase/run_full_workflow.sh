#!/bin/bash
# Complete RCDNet Workflow: 2023→2024 Year-Over-Year Change Detection
# Île-de-France, France - Tile T31UEQ

set -e  # Exit on error

export PYTHONPATH=/home/noam/referring_change_detection/RCDNet/models/encoders/selective_scan:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "════════════════════════════════════════════════════════════════════════════════"
echo "RCDNet Year-Over-Year Change Detection Workflow"
echo "Île-de-France, France - 2023 vs 2024"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Extract 2023 RGB composites
echo "Step 1: Extracting RGB composites from 2023 Sentinel-2 data..."
echo "────────────────────────────────────────────────────────────────────────────────"
python3 showcase/scripts/01_extract_sentinel.py \
    --input sentinel_data_2023 \
    --output showcase/data_2023/raw
echo ""
echo "✅ 2023 RGB extraction complete"
echo ""

# Step 2: Extract 2024 RGB composites (if not already done)
if [ ! -d "showcase/data/raw" ] || [ -z "$(ls -A showcase/data/raw/*.png 2>/dev/null)" ]; then
    echo "Step 2: Extracting RGB composites from 2024 Sentinel-2 data..."
    echo "────────────────────────────────────────────────────────────────────────────────"
    python3 showcase/scripts/01_extract_sentinel.py
    echo ""
    echo "✅ 2024 RGB extraction complete"
    echo ""
else
    echo "Step 2: 2024 RGB composites already extracted ✓"
    echo ""
fi

# Step 3: Create temporal pairs (2023 → 2024)
echo "Step 3: Creating temporal pairs (2023 → 2024)..."
echo "────────────────────────────────────────────────────────────────────────────────"
python3 showcase/scripts/02_create_pairs.py \
    --data-2023 showcase/data_2023/raw \
    --data-2024 showcase/data/raw \
    --output showcase/data_comparison
echo ""
echo "✅ Temporal pairs created"
echo ""

# Step 4: Run tiled inference (4GB GPU compatible)
echo "Step 4: Running change detection inference (tiled for 4GB GPU)..."
echo "────────────────────────────────────────────────────────────────────────────────"
python3 showcase/scripts/03_run_inference_tiled.py \
    --config configs.config_sentinel_showcase \
    --checkpoint weights/SECOND-model.safetensors \
    --output showcase/results_2023_2024 \
    --device cuda \
    --tile-size 256
echo ""
echo "✅ Inference complete"
echo ""

# Step 5: Visualize results
echo "Step 5: Creating visualizations..."
echo "────────────────────────────────────────────────────────────────────────────────"
python3 showcase/scripts/04_visualize.py \
    --input showcase/results_2023_2024 \
    --data showcase/data_comparison \
    --output showcase/visualizations
echo ""
echo "✅ Visualizations created"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "WORKFLOW COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  • Change maps: showcase/results_2023_2024/change_maps/"
echo "  • Statistics:  showcase/results_2023_2024/statistics/"
echo "  • Visuals:     showcase/visualizations/"
echo ""
echo "Expected changes detected:"
echo "  • Urban development (new buildings)"
echo "  • Infrastructure changes"
echo "  • Vegetation changes"
echo "  • Land use modifications"
echo ""
