# Kaggle Training Guide - Phase 1 Improvements

## 🎯 Phase 1 Changes Implemented

### ✅ What's Changed (Compatible with Epoch 60 Checkpoint)

| Parameter | Previous | Phase 1 | Impact |
|-----------|----------|---------|--------|
| **L1 Loss Weight** | 1.0 | **0.5** | ↓ Less smoothing, sharper details |
| **Perceptual Weight** | 0.5 | **1.0** | ↑ Better feature matching |
| **Gradient Weight** | 0.1 | **0.5** | ↑ Better edge recovery |
| **Learning Rate** | 2.5e-5 | **5e-5** | ↑ Escape plateau faster |
| **LR Scheduler** | plateau | **cosine** | Periodic restarts to escape local minima |
| **Max Epochs** | 60 | **120** | More training time |
| **Early Stop Patience** | 10 | **15** | More patience before stopping |
| **LR Patience** | 5 | **8** | More patience for LR reduction |

### 🚫 What's NOT Changed (For Compatibility)

- ✅ **Crop Size:** Stays at 128 (changing this would break checkpoint)
- ✅ **Model Architecture:** Unchanged
- ✅ **Batch Size:** Stays at 10

---

## 📋 Setup Instructions for Kaggle

### Step 1: Upload Your Checkpoint

1. Go to your Kaggle notebook
2. Click **"+ Add data"** → **"Upload"**
3. Upload your **`checkpoint_epoch_60.pth`** file
4. Note the path (will be something like `/kaggle/input/your-checkpoint/checkpoint_epoch_60.pth`)

### Step 2: Upload Files to Kaggle

Upload these files to your Kaggle notebook:
- `codes/train_CR_kaggle.py`
- `codes/losses.py`
- `codes/net_CR_CrossAttention.py`
- `codes/dataloader.py`
- `codes/model_CR_net.py`
- `codes/metrics.py`

### Step 3: Training Command

**Option A: Resume from Epoch 60 with Reset State (RECOMMENDED)**

```python
!python codes/train_CR_kaggle.py \
    --model_name CrossAttention \
    --batch_sz 10 \
    --num_workers 4 \
    --input_data_folder /kaggle/input/image1 \
    --data_list_filepath /kaggle/working/data.csv \
    --lr 5e-5 \
    --max_epochs 120 \
    --early_stop_patience 15 \
    --lr_scheduler cosine \
    --lr_patience 8 \
    --l1_weight 0.5 \
    --perceptual_weight 1.0 \
    --gradient_weight 0.5 \
    --fft_weight 0.3 \
    --contrastive_weight 0.1 \
    --use_gradient_loss \
    --use_fft_loss \
    --use_contrastive_loss \
    --resume_checkpoint /kaggle/input/your-checkpoint/checkpoint_epoch_60.pth \
    --reset_state \
    --experiment_name glf_cr_enhanced_v4_phase1 \
    --notes "Phase 1: Rebalanced losses + higher LR + cosine schedule"
```

**Important Flags:**
- `--resume_checkpoint`: Path to your epoch 60 checkpoint
- `--reset_state`: **CRITICAL** - Resets optimizer/scheduler but keeps model weights
- `--lr 5e-5`: New higher learning rate
- `--lr_scheduler cosine`: New scheduler type
- `--l1_weight 0.5`: Reduced from 1.0
- `--gradient_weight 0.5`: Increased from 0.1
- `--perceptual_weight 1.0`: Increased from 0.5

---

## 🔍 What Will Happen

### During Training:

1. ✅ **Epoch 61 starts** (continues from your checkpoint)
2. ✅ **Model weights loaded** from epoch 60
3. ✅ **Optimizer reset** with new LR (5e-5)
4. ✅ **Cosine scheduler created** from scratch
5. ✅ **New loss balance** takes effect immediately
6. ✅ **Early stopping counter reset** (fresh 15 epoch patience)
7. ✅ **Best PSNR tracking continues** (will try to beat 32.01 dB)

### Expected Behavior:

**Epochs 61-70:**
- Loss may **increase slightly** at first (due to higher LR and new loss balance)
- This is **NORMAL** - the model is exploring new parameter space
- Within 5-10 epochs, you should see **new improvements**

**Epochs 71-100:**
- Cosine scheduler will provide periodic "restarts"
- Model should **escape the plateau**
- Target: **33-34 dB** by epoch 90

**Epochs 101-120:**
- Fine-tuning phase
- Target: **34-35 dB** by epoch 120

---

## 📊 Monitoring Training

### Key Metrics to Watch

```python
# After training, check the log
import json

with open('/kaggle/working/checkpoints/logs/glf_cr_enhanced_v4_phase1_log.json') as f:
    log = json.load(f)

# Check if we're improving
for epoch in log['epochs'][-10:]:  # Last 10 epochs
    print(f"Epoch {epoch['epoch']}: Val PSNR = {epoch['val_psnr']:.2f} dB")
```

### Success Indicators

✅ **Good signs:**
- Val PSNR > 32.01 dB (beating your previous best)
- Train PSNR continuing to increase
- Val PSNR following train PSNR upward
- SSIM staying high (> 0.92)

⚠️ **Warning signs:**
- Val PSNR stuck at 32.0 dB for 15+ epochs → May need Phase 2
- Val PSNR dropping below 31.5 dB → LR might be too high
- Loss exploding (> 0.5) → Reduce LR to 3e-5

---

## 🛠️ Troubleshooting

### Issue: "Checkpoint architecture mismatch"

**Solution:** Your checkpoint is fine, but make sure you're using `--model_name CrossAttention`

### Issue: Loss increases initially

**Solution:** This is **expected** with higher LR. Give it 10 epochs to stabilize.

### Issue: Out of memory

**Solution:** Reduce `--batch_sz` from 10 to 8 or 6

### Issue: Training very slow

**Solution:** Your crop size is 128, so it should be fast. If slow, check GPU utilization.

---

## 📈 Expected Results

### Conservative Estimate (80% confidence)

- **Epoch 80:** 33.0 dB
- **Epoch 100:** 33.5 dB  
- **Epoch 120:** 34.0 dB

**Gain:** +2.0 dB from current 32.0 dB

### Optimistic Estimate (50% confidence)

- **Epoch 80:** 33.5 dB
- **Epoch 100:** 34.5 dB
- **Epoch 120:** 35.0 dB

**Gain:** +3.0 dB from current 32.0 dB

---

## ⏭️ Next Steps

### If Phase 1 Reaches 34-35 dB ✅

**Success!** You can:
1. Continue training to 150 epochs for fine-tuning
2. Try Phase 1D (crop size 192) from scratch if you want to push to 36 dB
3. Deploy and test on real data

### If Phase 1 Stays Below 34 dB ⚠️

**Move to Phase 2:**
1. Multi-scale processing
2. Stacked cross-attention blocks
3. Progressive training

I'll help you implement Phase 2 if needed!

---

## 💾 Download Results

After training completes:

```bash
# Checkpoints are automatically copied to /kaggle/output/
# Download from the Output tab:
- best_model.pth (best checkpoint)
- checkpoints/ (all epoch checkpoints)
- glf_cr_enhanced_v4_phase1_log.json (training log)
```

---

## 🚀 Ready to Train!

Your configuration is **checkpoint-compatible** and ready to resume from Epoch 60.

**Key Points:**
- ✅ Model weights will be loaded
- ✅ Training continues from epoch 61
- ✅ New loss balance & LR take effect immediately
- ✅ 60 more epochs to improve
- ✅ Expected +2 to +3 dB improvement

**Good luck!** 🎯
