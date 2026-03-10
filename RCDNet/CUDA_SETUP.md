# CUDA Setup for RCDNet Selective Scan Module

## Current Status

✅ **GPU Detected**: NVIDIA GeForce GTX 1650
✅ **CUDA Runtime**: 12.6
❌ **CUDA Toolkit**: Not installed (nvcc missing)

## Why It's Needed

RCDNet uses **VMamba** backbone which requires the **selective_scan** CUDA extension for efficient state space model operations. This module must be compiled from source using CUDA.

---

## Installation Options

### Option 1: Install CUDA Toolkit (Recommended)

Install CUDA 12.x toolkit that matches your runtime version:

```bash
# Download and install CUDA Toolkit 12.6
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# Add to PATH (add to ~/.bashrc for permanence)
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

Then build selective_scan:

```bash
cd /home/noam/referring_change_detection/RCDNet/models/encoders/selective_scan
pip install -e .
cd ../../..
```

### Option 2: Use Conda CUDA Toolkit (Easier)

Install CUDA toolkit through conda:

```bash
conda install -c nvidia cuda-toolkit=12.6 -y

# Build selective_scan
cd models/encoders/selective_scan
pip install -e .
cd ../../..
```

### Option 3: Docker (Most Reliable)

Use pre-built Docker environment with CUDA:

```bash
# Pull PyTorch image with CUDA 12.1
docker pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Run container with GPU access
docker run --gpus all -it \
    -v /home/noam/referring_change_detection/RCDNet:/workspace \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Inside container
cd /workspace/models/encoders/selective_scan
pip install -e .
```

---

## Verification

After installation, verify selective_scan is working:

```bash
# Set LD_LIBRARY_PATH to include PyTorch libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Verify import
python3 -c "import selective_scan_cuda_core; print('✅ selective_scan installed successfully')"
```

**Important**: Add this to your `~/.bashrc` for permanent effect:
```bash
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Alternative: CPU-Only Mode (Not Recommended)

If CUDA compilation continues to fail, you can try running inference on CPU:

**Pros**:
- No CUDA toolkit needed
- Simpler setup

**Cons**:
- **Much slower** (~10-50x slower than GPU)
- May not work due to CUDA-specific operations in selective_scan

**To attempt CPU mode**:

You would need to modify the selective_scan module to use CPU implementations, which is beyond the scope of this guide. **GPU is highly recommended.**

---

## Troubleshooting

### Error: "nvcc not found"
- Install CUDA toolkit (Option 1 or 2 above)
- Check PATH: `echo $PATH | grep cuda`

### Error: "CUDA version mismatch"
- Your CUDA toolkit version should match or be compatible with CUDA runtime
- Check runtime: `nvidia-smi` (shows CUDA 12.6)
- Check toolkit: `nvcc --version`

### Error: "undefined symbol" during import
- Rebuild with correct CUDA version
- Clean build: `cd models/encoders/selective_scan && rm -rf build/ && pip install -e .`

### WSL-Specific Issues
- Ensure WSL2 is being used (not WSL1)
- GPU passthrough must be enabled
- Install CUDA for WSL: https://docs.nvidia.com/cuda/wsl-user-guide/

---

## Next Steps

Once selective_scan is built, you can run the full inference pipeline:

```bash
# Test on 10 patches
python3 showcase/scripts/03_run_inference.py \
    --checkpoint weights/SECOND-model.safetensors \
    --max-patches 10 \
    --device cuda

# Full inference on all 841 patches
python3 showcase/scripts/03_run_inference.py \
    --checkpoint weights/SECOND-model.safetensors \
    --device cuda \
    --output showcase/results
```

**Estimated time**: 10-15 minutes on GTX 1650 for 841 patches

---

## References

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [CUDA WSL Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)
- [VMamba GitHub](https://github.com/MzeroMiko/VMamba)
