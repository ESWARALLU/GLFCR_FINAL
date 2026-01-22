#!/bin/bash
# Quick installation script for GLF-CR with modern PyTorch
# Run with: bash install.sh

set -e  # Exit on error

echo "========================================"
echo "GLF-CR Installation Script"
echo "Updated for PyTorch 2.x / CUDA 12.x"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Environment name
ENV_NAME="glf-cr-updated"

echo ""
echo "Step 1: Creating conda environment..."
conda create -n $ENV_NAME python=3.9 -y

echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "Step 3: Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "✅ Found CUDA version: $CUDA_VERSION"
    
    # Determine PyTorch CUDA version to install
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        TORCH_CUDA="cu121"
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        TORCH_CUDA="cu118"
    else
        echo "⚠️  Unusual CUDA version. Defaulting to cu118"
        TORCH_CUDA="cu118"
    fi
else
    echo "⚠️  nvcc not found. Defaulting to CUDA 11.8"
    TORCH_CUDA="cu118"
fi

echo ""
echo "Step 4: Installing PyTorch with CUDA $TORCH_CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CUDA

echo ""
echo "Step 5: Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 6: Compiling CUDA extension..."
cd codes/FAC/kernelconv2d/
python setup.py clean
python setup.py install --user
cd ../../..

echo ""
echo "Step 7: Verifying installation..."
python -c "
import torch
import sys

print('=== Verification ===')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ Installation successful!')
else:
    print('⚠️  CUDA not available')
"

echo ""
echo "========================================"
echo "✅ Installation complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the model:"
echo "  python codes/test_CR.py"
echo ""
