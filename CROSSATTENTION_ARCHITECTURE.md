# Cross-Attention Cloud Removal Architecture

## Overview
This document describes the new **CloudRemovalCrossAttention** architecture, which replaces the previous RDN-based model with a more effective dual-encoder approach using global cross-attention and speckle-aware gating.

## Architecture Components

### 1. **Dual Encoders**
- **OpticalEncoder**: Processes the cloudy optical image (3 channels)
  - E1: Conv blocks → (B, 64, H, W)
  - E2: Downsampled → (B, 128, H/2, W/2)
  - E3: Downsampled → (B, 256, H/4, W/4)

- **SAREncoder**: Processes the co-registered SAR image (1 channel)
  - Same architecture structure, optimized for 1-channel SAR input
  - Produces feature maps at matching scales

### 2. **Global Cross-Attention Fusion**
- Takes optical features as **Query** and SAR features as **Key/Value**
- SAR information guides the refinement of optical features
- Applied at the deepest level (E3 scale: 256 channels)
- Multi-head attention mechanism for learning multiple relationships

### 3. **Speckle-Aware Gating Module**
- **GateNet**: Simple 1x1 convolution pathway on SAR features
- Generates attention gates: `σ(feat_sar)` 
- Gates the cross-attention output: `cross_out ⊙ σ`
- Mitigates SAR speckle noise in the refined features

### 4. **Refinement Block**
- Concatenates gated output with original optical E3 features
- Refines the fused representation before decoding

### 5. **Symmetric Decoder**
- **UpLevel-1**: Upsample (H/4, W/4) → (H/2, W/2)
  - Concatenate with optical E2 features (skip connection)
  - Process through ConvBlock → (B, 128, H/2, W/2)

- **UpLevel-2**: Upsample (H/2, W/2) → (H, W)
  - Concatenate with optical E1 features (skip connection)
  - Process through ConvBlock → (B, 64, H, W)

- **Final Output**: 1x1 convolution → (B, 3, H, W) cloud-free image

## Data Flow
```
Optical (B, 3, H, W)  ─────────────────────────┐
                                               ├→ OpticalEncoder → [E1, E2, E3]
                                               │
                    ┌──────────────────────────┘
                    │
SAR (B, 1, H, W) ───┤
                    │
                    └──────────────────────────┬→ SAREncoder → [E1, E2, E3]
                                               │
                    ┌──────────────────────────┘
                    │
                    ├→ CrossAttention(E3_opt, E3_sar)
                    │     ↓
                    ├→ SpeckleGating(cross_out, E3_sar)
                    │     ↓
                    ├→ Refinement(gated, E3_opt)
                    │     ↓
                    ├→ Decoder with skip connections
                    │     ↓
                    └→ Output (B, 3, H, W)
```

## Model Parameters
- **Total Parameters**: ~2.5M (optimized)
- **Key Hyperparameters**:
  - `num_heads`: 8 (multi-head attention)
  - `dim`: 256 (deepest feature dimension)
  - Activation: ReLU (encoders), GELU-compatible design

## Usage

### Training with CrossAttention
```bash
cd codes/

# Train with new CrossAttention architecture
python train_CR.py \
    --model_name CrossAttention \
    --batch_sz 4 \
    --max_epochs 100 \
    --lr 1e-4 \
    --crop_size 128

# Or use Kaggle version
python train_CR_kaggle.py \
    --model_name CrossAttention \
    --batch_sz 4 \
    --max_epochs 50
```

### Testing with CrossAttention
```bash
# Test with new model
python test_CR.py \
    --model_type CrossAttention \
    --checkpoint_path ../ckpt/CR_net.pth

# Or Kaggle version
python test_CR_kaggle.py \
    --model_type CrossAttention \
    --checkpoint_path /path/to/checkpoint.pth
```

### Using CrossAttention Model Directly
```python
from net_CR_CrossAttention import CloudRemovalCrossAttention

# Create model
model = CloudRemovalCrossAttention().cuda()

# Forward pass
optical_img = torch.randn(4, 3, 256, 256)  # Cloudy optical
sar_img = torch.randn(4, 1, 256, 256)      # SAR image

output = model(optical_img, sar_img)  # (4, 3, 256, 256)
```

## Key Differences from RDN

| Aspect | RDN | CrossAttention |
|--------|-----|-----------------|
| **Encoder** | Single encoder | Dual encoders (Optical + SAR) |
| **Feature Fusion** | Late concatenation | Global cross-attention |
| **SAR Integration** | Concatenation only | Query-Key-Value attention + gating |
| **Noise Handling** | General convolutions | Speckle-aware gating |
| **Skip Connections** | Within RDB blocks | Symmetric decoder-only |
| **Parameters** | ~7M | ~2.5M |

## Training Considerations

1. **Data Preparation**:
   - Ensure optical and SAR images are co-registered
   - Normalize to [0, 1] range
   - Match spatial dimensions

2. **Loss Function**:
   - Default: L1 Loss
   - Alternative: Can use L2 + SSIM for better perceptual quality
   - Modify in wrapper class if needed

3. **Learning Rate Schedule**:
   - Initial: 1e-4
   - Decay: StepLR with step_size=5, gamma=0.5
   - Adjustable in training scripts

4. **Batch Size**:
   - Recommended: 4-8 (depends on GPU memory)
   - Minimum: 1 (for testing)

## Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index

## File Structure
```
codes/
├── net_CR_CrossAttention.py    # New architecture
├── train_CR.py                  # Updated with --model_name arg
├── train_CR_kaggle.py           # Updated with --model_name arg
├── test_CR.py                   # Updated with --model_type arg
├── test_CR_kaggle.py            # Updated with --model_type arg
├── net_CR_RDN.py               # Original RDN model (preserved)
└── ... (other files unchanged)
```

## Backward Compatibility
- Original RDN model remains fully functional
- Both models can coexist and be trained/tested independently
- Use `--model_name RDN` or `--model_name CrossAttention` to switch

## Next Steps
1. Train model with appropriate learning rates
2. Monitor PSNR/SSIM on validation set
3. Experiment with attention head counts and dimensions
4. Compare results with RDN baseline
5. Fine-tune hyperparameters for your dataset

## References
- Paper Concept: Multi-modal cloud removal using SAR guidance
- Architecture Inspired: Vision Transformers (timm library)
- Attention Mechanism: Standard multi-head cross-attention from Transformers
