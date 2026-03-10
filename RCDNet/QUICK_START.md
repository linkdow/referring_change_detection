# RCDNet Sentinel-2 Showcase - Quick Start Guide

**Last Updated**: 2026-02-01

## 🎯 Current Status

✅ **Data Prepared**: 841 Sentinel-2 image pairs ready
✅ **Scripts Complete**: All 4 pipeline scripts implemented
✅ **Pretrained Weights**: SECOND & CNAM-CD models available
✅ **selective_scan**: CUDA module built successfully!

---

## 🚀 Quick Start

**First, set up the environment** (add to ~/.bashrc for permanent effect):
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### 1. Run Inference

```bash
# Test on 10 patches (~1-2 minutes)
python3 showcase/scripts/03_run_inference.py \
    --checkpoint weights/SECOND-model.safetensors \
    --max-patches 10 \
    --device cuda

# Full run on all 841 patches (~10-15 minutes)
python3 showcase/scripts/03_run_inference.py \
    --checkpoint weights/SECOND-model.safetensors \
    --device cuda
```

### 2. Create Visualizations

```bash
python3 showcase/scripts/04_visualize.py \
    --data showcase/data \
    --results showcase/results \
    --num-samples 6
```

---

## 📊 Dataset Details

**Temporal Coverage**:
- Before: June 23, 2024
- After: June 25, 2024
- Interval: 2 days

**Spatial Coverage**:
- Region: Île-de-France (T31UEQ tile)
- Area: ~220 km²
- Resolution: 10m/pixel
- Patches: 841 @ 512×512 pixels

**Data Quality**:
- Processing Level: L2A (atmospherically corrected)
- Cloud Cover: <20%
- Bands: RGB (B02, B03, B04)

---

## 🎨 Change Detection Classes

The model detects changes in 7 semantic classes:

1. **Non-change** (background)
2. **Low Vegetation** (grasslands, crops, meadows)
3. **Non-vegetated Ground Surface** (bare soil, construction sites)
4. **Tree** (forests, urban parks, wooded areas)
5. **Water** (Seine river, lakes, ponds)
6. **Building** (urban development, construction)
7. **Playground** (sports fields, recreational areas)

---

## 📂 Output Structure

After running the pipeline:

```
showcase/
├── data/
│   ├── A/                         # 841 before images
│   ├── B/                         # 841 after images
│   └── pairs.txt
│
├── results/
│   ├── change_maps/               # Per-class change detection
│   │   ├── building/              # 841 binary masks
│   │   ├── water/
│   │   ├── tree/
│   │   ├── low_vegetation/
│   │   ├── non_vegetated_ground_surface/
│   │   └── playground/
│   ├── composite/                 # 841 multi-class RGB overlays
│   └── statistics/
│       └── change_statistics.json
│
└── visualizations/
    ├── before_after_comparison.png    # Side-by-side grid
    ├── class_breakdown.png            # Bar charts
    ├── change_heatmap.png             # Aggregate intensity
    ├── per_class/                     # Individual samples
    │   ├── building_samples.png
    │   ├── water_samples.png
    │   └── ...
    └── summary_report.txt             # Text summary
```

---

## 📈 Expected Results

Based on the 2-day temporal interval, expect:

**High Detection**:
- Construction starts/stops
- Water body changes (rainfall, drainage)
- Rapid vegetation changes (mowing, harvesting)

**Low Detection**:
- Gradual vegetation growth
- Slow urban development
- Seasonal changes

**Note**: A 2-day interval is quite short for change detection. For better results, consider:
- Longer intervals (weeks/months)
- Multiple time points
- Seasonal comparisons

---

## 🔧 Troubleshooting

### Inference Errors

**"CUDA out of memory"**:
```bash
# Reduce batch size
python3 showcase/scripts/03_run_inference.py \
    --batch-size 1 \
    --checkpoint weights/SECOND-model.safetensors
```

**"selective_scan not found"**:
```bash
# Rebuild selective_scan
cd models/encoders/selective_scan
python3 setup.py build_ext --inplace
cd ../../..
```

### Visualization Errors

**"Statistics file not found"**:
- Run inference first (03_run_inference.py)
- Check that `showcase/results/statistics/change_statistics.json` exists

**"No composite images found"**:
- Ensure inference completed successfully
- Check `showcase/results/composite/` directory

---

## 💡 Tips for Better Results

### 1. Optimize Inference Speed

```bash
# Use mixed precision (faster, less memory)
python3 showcase/scripts/03_run_inference.py \
    --checkpoint weights/SECOND-model.safetensors \
    --batch-size 8 \
    --device cuda
```

### 2. Focus on Specific Classes

Edit `configs/config_sentinel_showcase.py` to comment out classes you don't need:

```python
C.class_names = [
    "Non-change",
    # "Low Vegetation",        # Commented out
    # "Non-vegetated Ground Surface",
    "Tree",                     # Keep only these
    "Water",
    "Building",
    # "Playground",
]
```

### 3. Adjust Detection Threshold

```bash
# Lower threshold = more sensitive (more false positives)
# Higher threshold = less sensitive (fewer false positives)
python3 showcase/scripts/03_run_inference.py \
    --threshold 0.3 \  # Default: 0.5
    --checkpoint weights/SECOND-model.safetensors
```

---

## 📚 Additional Resources

### Documentation
- `SHOWCASE_PLAN.md` - Complete implementation plan
- `CUDA_SETUP.md` - CUDA installation guide
- `showcase/README.md` - Showcase overview

### Scripts
- `showcase/scripts/01_extract_sentinel.py` - Data extraction
- `showcase/scripts/02_create_pairs.py` - Temporal pairing
- `showcase/scripts/03_run_inference.py` - Change detection
- `showcase/scripts/04_visualize.py` - Visualization

### External Links
- [RCDNet Paper](https://github.com/...)
- [Sentinel-2 Documentation](https://sentinels.copernicus.eu/web/sentinel/sentinel-2)
- [SECOND Dataset](https://captain-whu.github.io/SCD/)
- [Copernicus Data Space](https://dataspace.copernicus.eu/)

---

## ✅ Verification Checklist

Before running inference, verify:

- [ ] selective_scan built successfully
- [ ] CUDA available (`nvidia-smi`)
- [ ] 841 image pairs in `showcase/data/A/` and `showcase/data/B/`
- [ ] Pretrained weights in `weights/SECOND-model.safetensors`
- [ ] Config file exists: `configs/config_sentinel_showcase.py`

After running inference, check:

- [ ] Results directory has 841 × 6 class masks (5,046 files)
- [ ] Composite directory has 841 RGB overlays
- [ ] Statistics JSON exists with metrics
- [ ] No error messages in console output

After visualization, verify:

- [ ] Before/after comparison grid created
- [ ] Class breakdown charts generated
- [ ] Change heatmap created
- [ ] Per-class samples exist
- [ ] Summary report generated

---

**Ready to run!** Once selective_scan builds successfully, execute the Quick Start commands above. 🚀
