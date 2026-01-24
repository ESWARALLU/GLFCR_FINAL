# GLF-CR Enhanced Architecture - v3.0

## Changes Implemented (2026-01-24)

### 🎯 Goal: Push PSNR from 34.6 dB → 35-36 dB

---

## Enhancements Added

### 1. Spectral Attention for SAR ✅
**File**: `net_CR_CrossAttention.py`

**New Module**: `SpectralAttentionSAR`
```python
class SpectralAttentionSAR(nn.Module):
    """
    Channel attention for SAR (VV, VH channels).
    Learns to weight different SAR channels adaptively.
    """
```

**Integration**:
- Added to `SAREncoder` before first ResBlock
- Automatically weights VV/VH channels based on content
- Parameters: ~96 (negligible)
- Expected gain: +0.2-0.3 dB

---

### 2. Cloud-Aware Weighted Residual ✅  
**File**: `net_CR_CrossAttention.py`

**New Module**: `CloudConfidencePredictor`
```python
class CloudConfidencePredictor(nn.Module):
    """
    Predicts cloud confidence map from bottleneck features.
    Output: 0 = clear, 1 = cloudy
    """
```

**Forward Pass** (Line 407-423):
```python
# Predict cloud confidence
cloud_confidence = self.cloud_predictor(refined_3)
cloud_confidence = F.interpolate(cloud_confidence, size=optical_img.shape[2:])

# Adaptive weighting:
# Clear (conf=0): output = input (preserved)
# Cloudy (conf=1): output = input + residual (corrected)
output = optical_img + cloud_confidence * residual_cloud
```

**Key Benefits**:
- ✅ Restored global residual (your original intent)
- ✅ Smart weighting: clear regions preserved, cloudy regions corrected
- ✅ Network focuses computation on cloudy areas
- Parameters: ~368K
- Expected gain: +0.3-0.5 dB

---

### 3. Cloud-Focused Loss Function ✅
**File**: `losses.py`

**New Class**: `CloudFocusedLoss`
```python
class CloudFocusedLoss(nn.Module):
    """
    Weighted loss that emphasizes cloudy regions.
    cloud_weight = 1 + factor * cloud_intensity
    """
```

**Usage** (Optional):
```bash
--use_cloud_focused_loss \
--cloud_weight_factor 2.0
```

**Benefits**:
- Higher loss weight on cloudy pixels
- Helps network focus on difficult regions
- Expected gain: +0.2-0.4 dB

---

## Architecture Summary

### Parameter Count:
```
Base (before):           5.63M
+ Spectral Attention:    +0.0001M
+ Cloud Predictor:       +0.37M
======================================
Total (enhanced):        6.00M (+6.6%)
```

### Memory:
- FP32: ~24 MB
- FP16 (AMP): ~12 MB
- Same batch size capacity as before

---

## Expected Performance

### Cumulative Gains:
```
Base architecture:                34.6 dB
+ Spectral SAR Attention:         +0.2-0.3 dB
+ Cloud-Aware Weighting:          +0.3-0.5 dB
+ Cloud-Focused Loss (optional):  +0.2-0.4 dB
============================================
Expected total:                   35.3-36.2 dB ✅
```

---

## Training Command (Updated)

```bash
!python train_CR_kaggle.py \
  --model_name CrossAttention \
  --max_epochs 30 \
  --batch_sz 6 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --lr_scheduler plateau \
  --lr_patience 5 \
  --early_stop_patience 10 \
  --grad_clip 1.0 \
  --warmup_epochs 2 \
  --use_cross_attn \
  --use_fft_loss --fft_weight 0.3 \
  --use_gradient_loss --gradient_weight 0.2 \
  --use_contrastive_loss --contrastive_weight 0.1 \
  --perceptual_weight 0.5 \
  --l1_weight 1.0 \
  --use_cloud_focused_loss \
  --cloud_weight_factor 2.5 \
  --data_list_filepath /kaggle/working/data.csv \
  --input_data_folder /kaggle/input/image1 \
  --save_model_dir /kaggle/working/checkpoints \
  --experiment_name glf_cr_enhanced_v3 \
  --train_from_scratch \
  --notes "Enhanced: Spectral attention + cloud-aware weighting + focused loss"
```

---

## Files Modified

1. **net_CR_CrossAttention.py**:
   - Added `SpectralAttentionSAR` class
   - Modified `SAREncoder` to use spectral attention
   - Added `CloudConfidencePredictor` class
   - Modified `CloudRemovalCrossAttention.forward()` for cloud-aware weighting
   - Restored global residual connection with adaptive weighting

2. **losses.py**:
   - Added `CloudFocusedLoss` class

3. **train_CR_kaggle.py**:
   - Added `--use_cloud_focused_loss` argument
   - Added `--cloud_weight_factor` argument

---

## Testing

Run model test:
```python
python net_CR_CrossAttention.py
```

Expected output:
```
Model parameters: 6.00M
Architecture: Cloud-Aware Weighted Residual Learning
  - Spectral Attention for SAR: ✓
  - Cloud Confidence Predictor: ✓
  - Adaptive Residual Weighting: ✓
Expected PSNR target: 35-36 dB
```

---

## Next Steps

1. ✅ Changes implemented
2. ⏳ Push to GitHub
3. ⏳ Train and validate

**Target**: 35-36 dB PSNR @ epoch 10-15
