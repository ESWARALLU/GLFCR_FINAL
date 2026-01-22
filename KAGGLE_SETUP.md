# Kaggle Setup Guide for GLF-CR Training

## Steps to run on Kaggle Notebook:

### 1. Clone Repository and Install Dependencies
```bash
!git clone https://github.com/mohit-epic/glf-crPLUS.git
%cd glf-crPLUS/GLF-CR/codes
!pip install -r ../requirements.txt -q
```

### 2. Check GPU Availability
```bash
!nvidia-smi
```
You should see 2 T4 GPUs available

### 3. Testing with Pretrained Model (test_CR_kaggle.py)
```bash
# Test on full test set (15%)
!python test_CR_kaggle.py \
    --batch_sz 4 \
    --num_workers 4 \
    --input_data_folder "/kaggle/input/image1" \
    --data_list_filepath "/kaggle/input/image2/data.csv" \
    --checkpoint_path "/kaggle/input/image3/CR_net.pth" \
    --model_name "baseline_full_test" \
    --gpu_ids "0,1"

# Or test single batch
!python test_CR_kaggle.py \
    --batch_sz 4 \
    --num_workers 4 \
    --input_data_folder "/kaggle/input/image1" \
    --data_list_filepath "/kaggle/input/image2/data.csv" \
    --checkpoint_path "/kaggle/input/image3/CR_net.pth" \
    --model_name "baseline_single_batch" \
    --single_batch \
    --gpu_ids "0,1"
```

### 4. Training from Scratch (train_CR_kaggle.py)
```bash
!python train_CR_kaggle.py \
    --batch_sz 4 \
    --num_workers 4 \
    --max_epochs 10 \
    --save_freq 1 \
    --input_data_folder "/kaggle/input/image1" \
    --data_list_filepath "/kaggle/input/image2/data.csv" \
    --experiment_name "kaggle_training_v1" \
    --gpu_ids "0,1"
```

### 5. Resume Training from Checkpoint
```bash
# First, check if best_model.pth exists in image3
!python train_CR_kaggle.py \
    --batch_sz 4 \
    --num_workers 4 \
    --max_epochs 10 \
    --save_freq 1 \
    --input_data_folder "/kaggle/input/image1" \
    --data_list_filepath "/kaggle/input/image2/data.csv" \
    --resume_checkpoint "/kaggle/input/image3/best_model.pth" \
    --experiment_name "kaggle_resume_v1" \
    --gpu_ids "0,1"
```

### 6. Download Results
Results are saved to `/kaggle/working/results/`
- Individual run results: `result_*.json`
- Training logs: `checkpoints/logs/*.json`
- Checkpoints: `checkpoints/checkpoint_epoch_*.pth`

These will be available in the Output tab when notebook finishes.

## Key Changes for Kaggle:

1. **Multi-GPU Support**: Scripts use `--gpu_ids "0,1"` for 2 T4 GPUs
2. **Kaggle Paths**:
   - Input data: `/kaggle/input/image1/`
   - Data CSV: `/kaggle/input/image2/data.csv`
   - Models: `/kaggle/input/image3/`
   - Output: `/kaggle/working/`
3. **No Image Saving**: `test_CR_kaggle.py` only outputs PSNR/SSIM values
4. **Better Logging**: Per-image PSNR/SSIM values printed to console
5. **Checkpoint Compatibility**: Handles both old and new checkpoint formats

## Tips:

- For 9-hour sessions, train for 5-10 epochs max
- Use `--batch_sz 8` if you have memory issues (smaller batches)
- Increase `--num_workers` if data loading is slow
- Monitor GPU with `!nvidia-smi` in separate cell
