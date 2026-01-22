# Complete Execution Flow: test_CR.py
## Deep Dive into Every Function Call

## Command Executed
```bash
python test_CR.py --model_name "baseline_v1" --notes "Initial baseline run after CUDA setup"
```

---

## 📋 Complete Step-by-Step Execution Flow with Every Function Explained

### **PHASE 1: Script Initialization**

#### 1.1 Python Imports (Lines 1-11)
```python
import os, torch, argparse, datetime, json
from metrics import PSNR, SSIM
from dataloader import AlignedDataset, get_train_val_test_filelists
from net_CR_RDN import RDN_residual_CR
```
- **What happens**: Python loads all required modules
- **Key imports**:
  - `torch` → PyTorch deep learning framework
  - `metrics` → PSNR and SSIM calculation functions
  - `dataloader` → Data loading utilities
  - `net_CR_RDN` → The GLF-CR model architecture

---

### **PHASE 2: Entry Point & Argument Parsing**

#### 2.1 Script Entry (Line 146)
```python
if __name__ == "__main__":
    main()
```
- **What happens**: Python executes `main()` function
- **Location**: test_CR.py, line 146

#### 2.2 GPU Selection (Line 58)
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```
- **Input**: None
- **Process**: Sets GPU device 0 as visible
- **Output**: Environment variable set
- **Effect**: All CUDA operations will use GPU 0 (your RTX 4060)

#### 2.3 Argument Parser Setup (Lines 60-72)
```python
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='../data')
parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
parser.add_argument('--is_test', type=bool, default=True)
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--model_name', type=str, default='baseline')
parser.add_argument('--notes', type=str, default='')

opts = parser.parse_args()
```
- **Input**: Command line arguments `--model_name "baseline_v1" --notes "Initial baseline..."`
- **Process**: Parses arguments, fills in defaults for missing ones
- **Output**: `opts` object with:
  ```
  opts.batch_sz = 1
  opts.load_size = 256
  opts.crop_size = 128
  opts.input_data_folder = '../data'
  opts.data_list_filepath = '../data/data.csv'
  opts.is_test = True
  opts.is_use_cloudmask = False
  opts.cloud_threshold = 0.2
  opts.model_name = 'baseline_v1'  ← Your input
  opts.notes = 'Initial baseline run after CUDA setup'  ← Your input
  ```

---

### **PHASE 3: Model Loading**

#### 3.1 Model Architecture Creation (Line 74)
```python
CR_net = RDN_residual_CR(opts.crop_size).cuda()
```
- **Input**: `crop_size=128`
- **Process**: 
  1. Calls `RDN_residual_CR.__init__(128)` in net_CR_RDN.py
  2. Constructs entire GLF-CR architecture:
     - **Optical encoder**: 6 RDB blocks with dense connections
     - **SAR encoder**: 6 RDB blocks 
     - **WindowAttention**: 8x8 window self-attention (Swin-style)
     - **DFG fusion modules**: Dynamic filtering guided fusion
     - **Decoder**: Upsampling layers
  3. `.cuda()` moves model to GPU
- **Output**: Model on GPU with ~millions of parameters
- **Memory**: Allocates GPU memory for all model weights

#### 3.2 Checkpoint Loading (Lines 75-76)
```python
checkpoint = torch.load('../ckpt/CR_net.pth', weights_only=False)
CR_net.load_state_dict(checkpoint['network'])
```
- **Input**: Pretrained weights file `../ckpt/CR_net.pth`
- **Process**:
  1. Loads checkpoint dictionary from disk
  2. Extracts `checkpoint['network']` (trained weights)
  3. Loads weights into model architecture
- **Output**: Model with trained parameters
- **Effect**: Model is now ready for inference (not random initialization)

#### 3.3 Set Evaluation Mode (Lines 78-80)
```python
CR_net.eval()
for _, param in CR_net.named_parameters():
    param.requires_grad = False
```
- **Input**: Trained model
- **Process**:
  1. `eval()` → Disables dropout, uses running stats for batch norm
  2. `requires_grad=False` → Disables gradient computation
- **Output**: Model in inference mode
- **Effect**: Faster inference, no backpropagation needed

---

### **PHASE 4: Testing Execution**

#### 4.1 Print Header (Lines 82-84)
```python
print(f"\n{'='*60}")
print(f"Testing Model: {opts.model_name}")
print(f"{'='*60}\n")
```
- **Output to terminal**:
  ```
  ============================================================
  Testing Model: baseline_v1
  ============================================================
  ```

#### 4.2 Call Test Function (Line 86)
```python
avg_psnr, avg_ssim, results_per_image = test(CR_net, opts)
```
**This triggers the main testing logic (Lines 14-56):**

---

### **PHASE 5: Test Function Execution (test_CR.py, lines 14-56)**

#### 5.1 Load File List (Line 16)
```python
_, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
```
**Calls dataloader.py → `get_train_val_test_filelists()`:**
- **Input**: `'../data/data.csv'`
- **Process**: 
  1. Opens CSV file
  2. Reads each line
  3. Separates into train/val/test based on first column:
     - `1` → train_filelist
     - `2` → val_filelist
     - `3` → test_filelist (your test data)
- **Output**: Returns three lists
- **Your data (from data.csv)**:
  ```
  test_filelist = [
    ['3', 's1', 's2_cloudfree', 's2_cloudy', 'ROIs1970_fall_139_p839.tif']
  ]
  ```
  (Only 1 test image)

#### 5.2 Create Dataset (Line 18)
```python
data = AlignedDataset(opts, test_filelist)
```
**Calls dataloader.py → `AlignedDataset.__init__()`:**
- **Input**: opts + test_filelist (1 image)
- **Process**: 
  1. Stores filelist
  2. Sets normalization parameters:
     - SAR: clip to [-25.0, 0], normalize to [0, 1]
     - Optical: clip to [0, 10000], divide by 10000
  3. Creates dataset object
- **Output**: Dataset object with 1 image (`self.n_images = 1`)

#### 5.3 Create DataLoader (Line 20)
```python
dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz, shuffle=False)
```
- **Input**: Dataset + batch_size=1
- **Process**: Wraps dataset in PyTorch DataLoader
- **Output**: Iterable dataloader
- **Effect**: Will load data in batches (batch_size=1 means 1 image at a time)

#### 5.4 Initialize Accumulators (Lines 22-25)
```python
iters = 0
total_psnr = 0
total_ssim = 0
results_per_image = []
```
- **Purpose**: Track metrics across all images

#### 5.5 Main Testing Loop (Lines 27-50)
```python
for inputs in dataloader:
```

**For each iteration (you have 1 image, so runs once):**

##### 5.5.1 Load Batch Data (Lines 29-32)
```python
cloudy_data = inputs['cloudy_data'].cuda()
cloudfree_data = inputs['cloudfree_data'].cuda()
SAR_data = inputs['SAR_data'].cuda()
file_name = inputs['file_name'][0]
```

**This triggers `AlignedDataset.__getitem__(index=0)` in dataloader.py:**

**Step A: Build File Paths (Lines 45-47)**
```python
fileID = self.filelist[0]  # ['3', 's1', 's2_cloudfree', 's2_cloudy', 'ROIs1970_fall_139_p839.tif']

s1_path = '../data/s1/ROIs1970_fall_139_p839.tif'
s2_cloudfree_path = '../data/s2_cloudfree/ROIs1970_fall_139_p839.tif'
s2_cloudy_path = '../data/s2_cloudy/ROIs1970_fall_139_p839.tif'
```

**Step B: Load Images from Disk (Lines 48-50)**
```python
s1_data = self.get_sar_image(s1_path).astype('float32')
s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')
```
- **Process**: Calls `rasterio.open()` to read GeoTIFF files
- **Output shapes**:
  - `s1_data`: (2, 256, 256) - SAR VV+VH polarizations
  - `s2_cloudfree_data`: (13, 256, 256) - Sentinel-2 13 bands (ground truth)
  - `s2_cloudy_data`: (13, 256, 256) - Sentinel-2 13 bands (cloudy input)

**Step C: Normalize Data (Lines 60-62)**
```python
s1_data = self.get_normalized_data(s1_data, data_type=1)  # SAR normalization
s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)  # Optical
s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)  # Optical
```
**In `get_normalized_data()`:**
- **SAR (type=1)**: 
  - Clip to [-25, 0]
  - Shift by +25 → [0, 25]
  - Normalize to [0, 1]
- **Optical (type=2/3)**:
  - Clip to [0, 10000]
  - Divide by 10000 → [0, 1]
- **Output**: All data in range [0, 1]

**Step D: Convert to Tensors (Lines 64-66)**
```python
s1_data = torch.from_numpy(s1_data)
s2_cloudfree_data = torch.from_numpy(s2_cloudfree_data)
s2_cloudy_data = torch.from_numpy(s2_cloudy_data)
```
- **Output**: PyTorch tensors

**Step E: Center Crop (Lines 68-76)**
```python
if self.opts.load_size - self.opts.crop_size > 0:  # 256 - 128 = 128 > 0
    if not self.opts.is_test:  # False (we're testing)
        # Random crop for training
    else:  # TRUE - we take center crop
        y = (256 - 128) // 2 = 64
        x = (256 - 128) // 2 = 64
    
    s1_data = s1_data[..., 64:192, 64:192]  # Crop to 128x128
    s2_cloudfree_data = s2_cloudfree_data[..., 64:192, 64:192]
    s2_cloudy_data = s2_cloudy_data[..., 64:192, 64:192]
```
- **Why?**: Model trained on 128x128 patches
- **Output**: Center 128x128 region extracted

**Step F: Return Dictionary (Lines 77-81)**
```python
results = {
    'cloudy_data': s2_cloudy_data,      # (13, 128, 128)
    'cloudfree_data': s2_cloudfree_data, # (13, 128, 128) - ground truth
    'SAR_data': s1_data,                 # (2, 128, 128)
    'file_name': 'ROIs1970_fall_139_p839.tif'
}
return results
```

**Back in test() function after dataloader returns:**
```python
cloudy_data = inputs['cloudy_data'].cuda()      # Move to GPU → (1, 13, 128, 128)
cloudfree_data = inputs['cloudfree_data'].cuda() # Move to GPU → (1, 13, 128, 128)
SAR_data = inputs['SAR_data'].cuda()            # Move to GPU → (1, 2, 128, 128)
file_name = 'ROIs1970_fall_139_p839.tif'
```
Note: Batch dimension added (1, ...) by DataLoader

##### 5.5.2 Model Inference (Line 34)
```python
pred_cloudfree_data = CR_net(cloudy_data, SAR_data)
```

**This triggers forward pass through RDN_residual_CR in net_CR_RDN.py:**

**Model Forward Pass Flow:**

1. **Optical Encoder** (cloudy_data → features)
   ```
   Input: (1, 13, 128, 128) cloudy optical image
   ↓
   Conv 3x3 → (1, 96, 128, 128)
   ↓
   RDB Block 1 → (1, 96, 128, 128)  [5 conv layers with dense connections]
   RDB Block 2 → (1, 96, 128, 128)
   RDB Block 3 → (1, 96, 128, 128)
   RDB Block 4 → (1, 96, 128, 128)
   RDB Block 5 → (1, 96, 128, 128)
   RDB Block 6 → (1, 96, 128, 128)
   ↓
   Global Feature Fusion → (1, 96, 128, 128)
   Output: opt_features
   ```

2. **SAR Encoder** (SAR_data → features)
   ```
   Input: (1, 2, 128, 128) SAR image
   ↓
   Conv 3x3 → (1, 96, 128, 128)
   ↓
   Same 6 RDB blocks as optical
   ↓
   Output: sar_features (1, 96, 128, 128)
   ```

3. **Window Attention** (on optical features)
   ```
   Input: opt_features (1, 96, 128, 128)
   ↓
   Reshape to (B, H, W, C)
   ↓
   Partition into 8x8 windows → (256 windows, 64 pixels per window, 96 channels)
   ↓
   Multi-head Self-Attention (8 heads):
     Q = Linear(opt_features)
     K = Linear(opt_features)
     V = Linear(opt_features)
     Attention = softmax(Q·K^T / √d) · V
   ↓
   Merge windows back
   ↓
   Output: attended_opt_features (1, 96, 128, 128)
   ```

4. **Dynamic Filtering Fusion (DFG)**
   ```
   Input: attended_opt_features + sar_features
   ↓
   Concatenate → (1, 192, 128, 128)
   ↓
   DFG Module:
     Predict dynamic kernels from concatenated features
     Kernels shape: (1, 96*9, 128, 128) [9 = 3x3 kernel size]
   ↓
   Apply KernelConv2D (custom CUDA kernel):
     Spatially-varying convolution using predicted kernels
     Each pixel gets unique 3x3 kernel
   ↓
   Output: fused_features (1, 96, 128, 128)
   ```

5. **Decoder** (fused_features → output)
   ```
   Input: fused_features (1, 96, 128, 128)
   ↓
   Upsampling Block:
     Conv → (1, 256, 128, 128)
     PixelShuffle(2x) → (1, 64, 128, 128)
   ↓
   Conv 3x3 → (1, 13, 128, 128)
   ↓
   Output: pred_cloudfree_data (1, 13, 128, 128)
   ```

- **Final Output**: Predicted cloud-free optical image (1, 13, 128, 128)

##### 5.5.3 Compute Metrics (Lines 36-37)
```python
psnr_13 = PSNR(pred_cloudfree_data, cloudfree_data)
ssim_13 = SSIM(pred_cloudfree_data, cloudfree_data).item()
```

**PSNR Calculation (metrics.py):**
```python
def PSNR(img1, img2, mask=None):
    mse = torch.mean((img1 - img2) ** 2)  # Mean squared error
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
```
- **Input**: pred (1,13,128,128), ground_truth (1,13,128,128)
- **Process**:
  1. Calculate MSE = mean((pred - truth)²)
  2. PSNR = 20 × log10(1 / √MSE)
- **Output**: `psnr_13 = 23.2544` dB

**SSIM Calculation (metrics.py):**
```python
def SSIM(img1, img2):
    # Creates 11x11 Gaussian window
    window = create_window(11, channel).cuda()
    
    # Compute local means using convolution
    mu1 = F.conv2d(img1, window, padding=5, groups=channel)
    mu2 = F.conv2d(img2, window, padding=5, groups=channel)
    
    # Compute variances and covariance
    sigma1_sq = ...
    sigma2_sq = ...
    sigma12 = ...
    
    # SSIM formula
    C1 = 0.01²
    C2 = 0.03²
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / 
               ((mu1² + mu2² + C1) * (sigma1² + sigma2² + C2))
    
    return ssim_map.mean()
```
- **Input**: Same pred and truth images
- **Process**: Structural similarity considering luminance, contrast, structure
- **Output**: `ssim_13 = 0.8084`

##### 5.5.4 Accumulate Results (Lines 39-47)
```python
total_psnr += psnr_13  # total_psnr = 23.2544
total_ssim += ssim_13  # total_ssim = 0.8084

results_per_image.append({
    'image': 'ROIs1970_fall_139_p839.tif',
    'psnr': 23.2544,
    'ssim': 0.8084
})

print(0, '  psnr_13:', '23.2544', '  ssim_13:', '0.8084')
iters += 1  # iters = 1
```
- **Terminal output**: `0   psnr_13: 23.2544   ssim_13: 0.8084`

#### 5.6 Calculate Averages (Lines 52-54)
```python
avg_psnr = total_psnr / iters  # 23.2544 / 1 = 23.2544
avg_ssim = total_ssim / iters  # 0.8084 / 1 = 0.8084

return avg_psnr, avg_ssim, results_per_image
```
- **Output**: Returns three values back to `main()`

---

### **PHASE 6: Results Display & Saving (Lines 88-145)**

#### 6.1 Print Summary (Lines 88-92)
```python
print(f"\n{'='*60}")
print(f"Average Results:")
print(f"  PSNR: {avg_psnr:.4f} dB")
print(f"  SSIM: {avg_ssim:.4f}")
print(f"{'='*60}\n")
```
- **Terminal output**:
  ```
  ============================================================
  Average Results:
    PSNR: 23.2544 dB
    SSIM: 0.8084
  ============================================================
  ```

#### 6.2 Create Results Directory (Lines 95-96)
```python
results_dir = '../results'
os.makedirs(results_dir, exist_ok=True)
```
- **Effect**: Creates `../results/` folder if doesn't exist

#### 6.3 Generate Timestamp (Line 98)
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Output: '20251120_193534'
```

#### 6.4 Build Result Entry (Lines 100-109)
```python
result_entry = {
    'timestamp': '20251120_193534',
    'datetime': '2025-11-20 19:35:34',
    'model_name': 'baseline_v1',
    'avg_psnr': 23.2544,
    'avg_ssim': 0.8084,
    'num_images': 1,
    'notes': 'Initial baseline run after CUDA setup',
    'per_image_results': [
        {'image': 'ROIs1970_fall_139_p839.tif', 'psnr': 23.2544, 'ssim': 0.8084}
    ]
}
```

#### 6.5 Save Individual Result File (Lines 112-114)
```python
result_file = '../results/result_baseline_v1_20251120_193534.json'
with open(result_file, 'w') as f:
    json.dump(result_entry, f, indent=4)
```
- **Creates**: `result_baseline_v1_20251120_193534.json`
- **Content**: Full JSON with all details

#### 6.6 Update History File (Lines 117-126)
```python
history_file = '../results/results_history.json'

# Load existing history
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)  # Load previous runs
else:
    history = []  # First run

# Append new result
history.append(result_entry)

# Save updated history
with open(history_file, 'w') as f:
    json.dump(history, f, indent=4)
```
- **Effect**: Appends this run to cumulative history

#### 6.7 Update CSV Summary (Lines 129-132)
```python
summary_file = '../results/results_summary.csv'
with open(summary_file, 'a') as f:  # Append mode
    if file_is_empty:
        f.write('Timestamp,Model,PSNR,SSIM,Num_Images,Notes\n')  # Header
    f.write('20251120_193534,baseline_v1,23.2544,0.8084,1,"Initial baseline run after CUDA setup"\n')
```
- **Effect**: Adds line to CSV for Excel viewing

#### 6.8 Print Confirmation (Lines 134-138)
```python
print(f"Results saved to:")
print(f"  - ../results/result_baseline_v1_20251120_193534.json")
print(f"  - ../results/results_history.json")
print(f"  - ../results/results_summary.csv\n")
```

---

## 📊 Complete Data Flow Summary

```
Command Line Arguments
    ↓
1. Parse Args → opts object
    ↓
2. Load Model Architecture → RDN_residual_CR (on GPU)
    ↓
3. Load Pretrained Weights → checkpoint['network']
    ↓
4. Set Eval Mode → No gradients
    ↓
5. Read data.csv → test_filelist (1 image)
    ↓
6. Create Dataset → AlignedDataset
    ↓
7. Create DataLoader → Batch iterator
    ↓
8. FOR EACH IMAGE (1 iteration):
    ↓
    8a. Load from disk → 3 GeoTIFF files
        - SAR: s1/ROIs1970_fall_139_p839.tif (2, 256, 256)
        - Ground truth: s2_cloudfree/ROIs1970_fall_139_p839.tif (13, 256, 256)
        - Input: s2_cloudy/ROIs1970_fall_139_p839.tif (13, 256, 256)
    ↓
    8b. Normalize → SAR [-25,0]→[0,1], Optical /10000
    ↓
    8c. Center Crop → 128x128 patches
    ↓
    8d. Move to GPU → .cuda()
    ↓
    8e. Model Forward Pass:
        - Optical Encoder (6 RDB blocks) → opt_features
        - SAR Encoder (6 RDB blocks) → sar_features
        - Window Attention → attended_opt_features
        - DFG Fusion (custom CUDA kernel) → fused_features
        - Decoder → pred_cloudfree (13, 128, 128)
    ↓
    8f. Compute PSNR → 23.2544 dB
    ↓
    8g. Compute SSIM → 0.8084
    ↓
    8h. Store results
    ↓
9. Calculate averages (same as single values for 1 image)
    ↓
10. Save to 3 files:
    - result_baseline_v1_20251120_193534.json
    - results_history.json (append)
    - results_summary.csv (append)
    ↓
11. Print confirmation
    ↓
END
```

---

## 🔧 Key Technical Details

### Memory Usage
- **Model parameters**: ~few million (exact count depends on architecture)
- **Input batch**: (1, 13, 128, 128) × 4 bytes = 851 KB
- **Intermediate features**: ~100-200 MB during forward pass
- **Total GPU memory**: ~1-2 GB for this model

### Computation
- **Operations per forward pass**: ~billions of FLOPs
- **Time per image**: ~0.1-0.5 seconds (depends on GPU)
- **Custom CUDA kernel**: KernelConv2D (compiled from .cu file)

### File I/O
- **Reads**: 3 GeoTIFF files (2D multi-channel raster images)
- **Writes**: 3 result files (2 JSON, 1 CSV)

---

## 🎯 What Makes This Model Unique

1. **Dual-stream encoder**: Processes optical (13 bands) and SAR (2 polarizations) separately
2. **Window-based attention**: Swin Transformer style 8×8 windows for efficiency
3. **Dynamic filtering**: Spatially-adaptive kernels predicted from features (KernelConv2D)
4. **Dense connections**: RDB blocks with feature reuse
5. **Multi-modal fusion**: SAR guides cloud removal in optical images

---

## 📈 Metrics Explained

**PSNR (Peak Signal-to-Noise Ratio)**
- Measures pixel-level reconstruction quality
- Range: typically 20-40 dB (higher is better)
- Your result: **23.25 dB** = reasonable cloud removal

**SSIM (Structural Similarity Index)**
- Measures perceptual quality (structure, luminance, contrast)
- Range: 0-1 (1 = perfect match)
- Your result: **0.8084** = good structural preservation

---

This is the complete execution flow from command to results!
