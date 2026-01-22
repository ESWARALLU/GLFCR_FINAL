# PowerShell installation script for GLF-CR on Windows
# Run with: .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GLF-CR Installation Script (Windows)" -ForegroundColor Cyan
Write-Host "Updated for PyTorch 2.x / CUDA 12.x" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit 1
}

# Environment name
$ENV_NAME = "glf-cr-updated"

Write-Host "`nStep 1: Creating conda environment..." -ForegroundColor Yellow
conda create -n $ENV_NAME python=3.9 -y

Write-Host "`nStep 2: Activating environment..." -ForegroundColor Yellow
conda activate $ENV_NAME

Write-Host "`nStep 3: Detecting CUDA version..." -ForegroundColor Yellow
$TORCH_CUDA = "cu121"  # Default to CUDA 12.1

if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $cudaVersion = nvcc --version | Select-String "release" | ForEach-Object { $_.ToString() -replace ".*release ", "" -replace ",.*", "" }
    Write-Host "✅ Found CUDA version: $cudaVersion" -ForegroundColor Green
    
    if ($cudaVersion -like "12.*") {
        $TORCH_CUDA = "cu121"
    } elseif ($cudaVersion -like "11.*") {
        $TORCH_CUDA = "cu118"
    }
} else {
    Write-Host "⚠️  nvcc not found. Defaulting to CUDA 12.1" -ForegroundColor Yellow
}

Write-Host "`nStep 4: Installing PyTorch with CUDA $TORCH_CUDA..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

Write-Host "`nStep 5: Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nStep 6: Compiling CUDA extension..." -ForegroundColor Yellow
Write-Host "⚠️  This may take a few minutes..." -ForegroundColor Yellow
Push-Location codes\FAC\kernelconv2d\
python setup.py clean
python setup.py install --user
Pop-Location

Write-Host "`nStep 7: Verifying installation..." -ForegroundColor Yellow
python -c @"
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
"@

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Yellow
Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
Write-Host ""
Write-Host "To test the model:" -ForegroundColor Yellow
Write-Host "  python codes\test_CR.py" -ForegroundColor White
Write-Host ""
