# GLF-CR Updated Requirements
# Compatible with modern CUDA versions (11.x, 12.x) and PyTorch 2.x

## Environment Setup

### Option 1: PyTorch 2.x with CUDA 12.x (Recommended)
```bash
conda create -n glf-cr-updated python=3.9
conda activate glf-cr-updated

# Install PyTorch 2.x with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR if you have CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: PyTorch 1.13.x with CUDA 11.x (More Compatible)
```bash
conda create -n glf-cr-updated python=3.9
conda activate glf-cr-updated

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Core Dependencies

```bash
# Scientific computing
pip install numpy>=1.19.0
pip install scipy>=1.5.0

# Image processing
pip install rasterio>=1.2.0
pip install pillow>=8.0.0

# Vision transformers (for enhancements)
pip install timm>=0.9.0

# Experiment tracking (optional)
pip install tensorboard
pip install wandb

# Utilities
pip install tqdm
pip install pyyaml
pip install matplotlib
pip install pandas

# For model analysis
pip install thop  # FLOPs computation
```

## CUDA Extension (KernelConv2D)

The original setup requires modifications for modern PyTorch/CUDA.

### Building the CUDA extension:
```bash
cd codes/FAC/kernelconv2d/
python setup.py clean
python setup.py install --user
```

If you encounter issues, see the updated `setup.py` with modern CUDA architectures.

## Verification

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Common Issues

### Issue 1: CUDA extension compilation fails
**Solution**: Update `setup.py` with modern CUDA architectures (see updated file)

### Issue 2: `torch.cuda.device_of` deprecated
**Solution**: Code has been updated to use `torch.cuda.device()`

### Issue 3: Mixed precision training issues
**Solution**: Use `torch.cuda.amp.autocast()` for modern mixed precision

## Summary

| Component | Original | Updated |
|-----------|----------|---------|
| Python | 3.6 | 3.9+ |
| PyTorch | 1.4.0 | 2.0+ or 1.13+ |
| CUDA | 10.1 | 11.7+ or 12.1+ |
| torchvision | 0.5.0 | 0.14+ or 0.15+ |
| timm | 0.3.2 | 0.9+ |
