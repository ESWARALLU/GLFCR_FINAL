"""
Kaggle Training Script - Phase 1 Improvements
Copy and paste this entire script into a Kaggle notebook cell
"""

# ========================================
# CELL 1: Clone Updated Repository
# ========================================
!git clone https://github.com/ESWARALLU/GLFCR_FINAL.git
%cd GLFCR_FINAL

# ========================================
# CELL 2: Install Dependencies
# ========================================
!pip install timm -q

# ========================================
# CELL 3: Setup Data CSV
# ========================================
# Copy your data.csv to working directory
# Adjust the path based on your Kaggle dataset setup
!cp /kaggle/input/your-dataset/data.csv /kaggle/working/data.csv

# ========================================
# CELL 4: Resume Training from Epoch 60
# ========================================
# IMPORTANT: Replace '/kaggle/input/your-checkpoint/checkpoint_epoch_60.pth' 
# with the actual path to your uploaded checkpoint

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
    --warmup_epochs 0 \
    --experiment_name glf_cr_enhanced_v4_phase1 \
    --notes "Phase 1: Rebalanced losses + higher LR + cosine schedule + resume from epoch 60"

# ========================================
# CELL 5: Monitor Results (After Training)
# ========================================
import json

# Load training log
with open('/kaggle/working/checkpoints/logs/glf_cr_enhanced_v4_phase1_log.json') as f:
    log = json.load(f)

# Print summary
print("="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Best Validation PSNR: {log['epochs'][-1]['best_val_psnr']:.2f} dB")
print(f"Previous Best: 32.01 dB")
print(f"Improvement: {log['epochs'][-1]['best_val_psnr'] - 32.01:.2f} dB")
print(f"Total Epochs: {len(log['epochs'])}")
print("="*60)

# Last 5 epochs
print("\nLast 5 Epochs:")
for epoch in log['epochs'][-5:]:
    print(f"Epoch {epoch['epoch']}: Train PSNR={epoch['train_psnr']:.2f}, Val PSNR={epoch['val_psnr']:.2f}, Val SSIM={epoch['val_ssim']:.4f}")
