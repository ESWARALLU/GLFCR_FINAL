# Kaggle Training & Download Guide

## Quick Start: Run Training on Kaggle

### Step 1: Run Training
Copy this into a Kaggle notebook cell:

```bash
!python codes/train_CR_kaggle.py \
    --batch_sz 8 \
    --num_workers 4 \
    --max_epochs 5 \
    --lr 1e-4 \
    --input_data_folder /kaggle/input/sen12ms-cr-winter \
    --data_list_filepath /kaggle/working/data.csv \
    --save_model_dir /kaggle/working/checkpoints \
    --experiment_name "my_training"
```

### Step 2: Automatic File Preparation

**NEW!** The training script now automatically copies checkpoints to `/kaggle/output` at the end of training.

You'll see output like:
```
✅ Best model copied to: /kaggle/output/best_model.pth
✅ All checkpoints copied to: /kaggle/output/checkpoints
✅ Training log copied to: /kaggle/output/training_log.json

📥 Download Instructions (Kaggle Notebooks)
============================================================
1. Click the 'Output' tab in your Kaggle notebook
2. You should see:
   - best_model.pth (best checkpoint)
   - checkpoints/ (all epoch checkpoints)
   - training_log.json (training logs)
3. Click the download icon next to each file
```

### Step 3: Download Files

1. **In your Kaggle notebook**, click the **Output** tab (right sidebar)
2. You'll see the downloaded files ready
3. Click the download icon next to each file

---

## Alternative: Manual Download (If Automatic Fails)

If the automatic copy doesn't work, use this helper:

```python
from codes.kaggle_checkpoint_helper import setup_kaggle_download

# Copy files to output directory
status = setup_kaggle_download(
    checkpoint_dir='/kaggle/working/checkpoints',
    output_dir='/kaggle/output',
    experiment_name='my_training',
    verbose=True
)

# Check if successful
if status['best_model']:
    print("✅ Files ready to download!")
```

---

## Find Your CPU Core Count

Add this cell to find optimal `num_workers`:

```python
import os
import multiprocessing

cpu_cores = os.cpu_count()
print(f"CPU cores available: {cpu_cores}")
print(f"Recommended num_workers: {min(cpu_cores * 2, 16)}")
```

**Typical Kaggle:**
- Free tier: 4 cores → use `num_workers=4-8`
- Pro tier: 8 cores → use `num_workers=8-16`

---

## Training Tips

### Batch Size & Workers Tuning

| Batch Size | Workers | Use Case |
|-----------|---------|----------|
| 4-8 | 4 | Stable training (RECOMMENDED) |
| 8 | 8 | Better speed, more memory |
| 16 | 8 | Faster training, less stable |
| 10 | 40 | ❌ TOO AGGRESSIVE |

**Rule:** `workers ≤ cpu_cores × 2`

### Learning Rate

Start with: `--lr 1e-4` (default)

If PSNR drops after 1 epoch:
- Try lower: `--lr 5e-5`
- If still bad: `--lr 1e-5`

If training is too slow:
- Try higher: `--lr 2e-4`
- Monitor for divergence

---

## Expected Output After Training

After each epoch:
```
============================================================
Epoch 0 Summary:
  Train Loss: 0.0383
  Train PSNR: 25.34 dB
  Val PSNR:   28.45 dB
  Val SSIM:   0.8234
  Best Val PSNR: 28.45 dB (NEW!)
  Epoch Time: 75.5 minutes
============================================================
```

After all epochs complete:
```
Training Complete!
============================================================
Total Time: 4.25 hours
Best Validation PSNR: 29.87 dB
Training log saved to: /kaggle/working/checkpoints/logs/...
Best model saved to: /kaggle/working/checkpoints/best_model.pth
============================================================

✅ Best model copied to: /kaggle/output/best_model.pth
✅ All checkpoints copied to: /kaggle/output/checkpoints
✅ Training log copied to: /kaggle/output/...

📥 Download Instructions (Kaggle Notebooks)
1. Click the 'Output' tab in your Kaggle notebook
2. You should see: best_model.pth, checkpoints/, ...
3. Click the download icon next to each file
```

---

## Load Checkpoint Locally

After downloading `best_model.pth`:

```python
import torch
from codes.model_CR_net import ModelCRNet

# Create model
opts = type('opts', (), {'other_params': ...})()
model = ModelCRNet(opts)

# Load checkpoint
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.net_G.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"   PSNR: {checkpoint['val_psnr']:.2f} dB")
```

---

## Troubleshooting

### Issue: Files not in Output tab
**Solution:** Run the helper cell after training:
```python
from codes.kaggle_checkpoint_helper import setup_kaggle_download
setup_kaggle_download()
```

### Issue: Training too slow
**Solution:** Reduce workers
```bash
--num_workers 4  # Instead of 8 or more
```

### Issue: PSNR too low
**Solution:** Check learning rate
```bash
--lr 1e-5  # Try lower LR
```

### Issue: Out of Memory
**Solution:** Reduce batch size or workers
```bash
--batch_sz 4 \
--num_workers 4
```

---

## Complete Example Command

```bash
!python codes/train_CR_kaggle.py \
    --batch_sz 8 \
    --num_workers 4 \
    --load_size 256 \
    --crop_size 128 \
    --max_epochs 10 \
    --lr 1e-4 \
    --lr_step 5 \
    --lr_start_epoch_decay 5 \
    --input_data_folder /kaggle/input/sen12ms-cr-winter \
    --data_list_filepath /kaggle/working/data.csv \
    --save_model_dir /kaggle/working/checkpoints \
    --experiment_name "GLF_CR_training_v1" \
    --notes "Training with optimized hyperparameters"
```

---

## Repository

All code available at: **https://github.com/ESWARALLU/glfCRPLUS**

Clone locally:
```bash
git clone https://github.com/ESWARALLU/glfCRPLUS.git
cd glfCRPLUS
```

---

**Last Updated:** December 13, 2025  
**Kaggle Notebook Compatibility:** ✅ Latest (2024+)  
**Status:** ✅ Production Ready
