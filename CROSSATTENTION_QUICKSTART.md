# CrossAttention Model - Quick Start Guide

## What's New?
A new **GlobalCrossAttention** architecture has been implemented alongside the existing RDN model. This architecture uses:
- Dual encoders (separate processing for Optical and SAR)
- Global cross-attention fusion (SAR guides optical feature refinement)
- Speckle-aware gating (reduces SAR speckle noise)
- Symmetric decoder with skip connections

## File Changes Summary

### New Files Created
- **net_CR_CrossAttention.py** - Complete implementation of the new architecture

### Modified Files
- **train_CR.py** - Added `--model_name` argument (RDN or CrossAttention)
- **train_CR_kaggle.py** - Added `--model_name` argument (RDN or CrossAttention)
- **test_CR.py** - Added `--model_type` argument (RDN or CrossAttention)
- **test_CR_kaggle.py** - Added `--model_type` argument (RDN or CrossAttention)

### Preserved Files
- **net_CR_RDN.py** - Original RDN implementation (unchanged)

## Quick Commands

### Train with CrossAttention (Local)
```bash
cd codes/
python train_CR.py --model_name CrossAttention --batch_sz 4 --max_epochs 100
```

### Train with CrossAttention (Kaggle)
```bash
python train_CR_kaggle.py --model_name CrossAttention --batch_sz 4 --max_epochs 50
```

### Test with CrossAttention
```bash
python test_CR.py --model_type CrossAttention --checkpoint_path ../ckpt/CR_net.pth
```

### Test with CrossAttention (Kaggle)
```bash
python test_CR_kaggle.py --model_type CrossAttention --checkpoint_path /path/to/checkpoint.pth
```

### Train with Original RDN (Backward Compatible)
```bash
python train_CR.py --model_name RDN --batch_sz 4 --max_epochs 100
# or just omit the --model_name arg (defaults to RDN)
python train_CR.py --batch_sz 4 --max_epochs 100
```

## Model Architecture Overview

```
INPUT:
├─ Optical Image (3, 256, 256)  → OpticalEncoder
└─ SAR Image (1, 256, 256)      → SAREncoder
           ↓
    [E1, E2, E3 features]
           ↓
   GlobalCrossAttention (E3 scale)
    SAR guides Optical
           ↓
   SpeckleAwareGating
    Reduces SAR speckle noise
           ↓
     Refinement Block
    Fuse with original optical
           ↓
     Symmetric Decoder
    UpLevel-1 (E3→E2 + skip)
    UpLevel-2 (E2→E1 + skip)
           ↓
OUTPUT: Cloud-free Image (3, 256, 256)
```

## Key Advantages

✅ **Dual Encoders**: Processes optical and SAR separately, learning modality-specific features

✅ **Cross-Attention**: SAR information directly guides optical feature refinement

✅ **Speckle-Aware Gating**: Mitigates SAR speckle noise using learned gating masks

✅ **Skip Connections**: Preserves fine details through decoder upsampling

✅ **Lighter Model**: ~2.5M parameters (vs ~7M for RDN)

✅ **Backward Compatible**: Existing RDN code continues to work unchanged

## Configuration Notes

The training wrapper class (`ModelCRNetCrossAttention`) provides compatibility with existing training loops:
- Uses L1 Loss by default (can be modified)
- Adam optimizer with StepLR scheduler
- Compatible with both single-GPU and multi-GPU training
- Supports checkpoint saving/loading

## Expected Input/Output Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| Optical Input | (B, 3, H, W) | Cloudy optical image |
| SAR Input | (B, 1, H, W) | Co-registered SAR image |
| Output | (B, 3, H, W) | Cloud-free optical image |
| E1 Features | (B, 64, H, W) | Encoder level 1 |
| E2 Features | (B, 128, H/2, W/2) | Encoder level 2 |
| E3 Features | (B, 256, H/4, W/4) | Encoder level 3 |

## Dataloader Compatibility

The updated training/test scripts expect dataloader outputs with these keys:
- `cloudy_optical`: (B, 3, H, W)
- `sar`: (B, 1, H, W)
- `cloudfree_optical`: (B, 3, H, W)
- `file_name`: List of filenames

Old field names are also supported for backward compatibility.

## Performance Tips

1. **Batch Size**: Use larger batches (8-16) if GPU memory allows for stable training
2. **Learning Rate**: Start with 1e-4, adjust down if loss oscillates
3. **Data Augmentation**: Add random flips/rotations to prevent overfitting
4. **Validation**: Monitor PSNR/SSIM every epoch, save best model
5. **Mixed Precision**: Can enable AMP for faster training (if needed)

## Troubleshooting

**Q: ModuleNotFoundError: torch**
A: Install dependencies: `pip install -r requirements.txt`

**Q: Model produces all black outputs**
A: Check data normalization is to [0, 1], not [-1, 1]

**Q: Out of memory errors**
A: Reduce batch size with `--batch_sz 2` or use smaller crop size `--crop_size 64`

**Q: Checkpoint incompatibility**
A: The wrapper handles different checkpoint formats - check file exists at specified path

## Next Steps

1. Install PyTorch and dependencies
2. Prepare your cloud removal dataset
3. Train with: `python train_CR.py --model_name CrossAttention`
4. Evaluate with: `python test_CR.py --model_type CrossAttention`
5. Compare metrics with RDN baseline
6. Fine-tune hyperparameters for your specific data

## Full Documentation
See `CROSSATTENTION_ARCHITECTURE.md` for detailed technical documentation.
