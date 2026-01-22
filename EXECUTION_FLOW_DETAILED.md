# COMPLETE EXECUTION FLOW: Deep Dive Analysis
## Every Function, Every Line, Every Transformation

## Command Executed
```bash
python test_CR.py --model_name "baseline_v1" --notes "Initial baseline run after CUDA setup"
```

---

# 🚀 EXECUTION TIMELINE

## PHASE 1: PYTHON INITIALIZATION (0.0s - 0.5s)

### 1.1 Module Import Chain

```python
import os
import torch
import argparse
from datetime import datetime
import json
```

**What happens internally:**
1. **os module**: Python's operating system interface
   - Provides `environ` for env variables
   - Provides `path.join`, `makedirs` for file operations
   
2. **torch module**: PyTorch deep learning framework
   - Loads CUDA runtime libraries (cublas, cudnn, etc.)
   - Initializes GPU context
   - Registers custom CUDA kernels
   - Memory: Allocates ~500MB for PyTorch internals
   
3. **argparse**: Command line argument parser
   - Creates ArgumentParser objects
   
4. **datetime**: Date/time utilities
   
5. **json**: JSON encoder/decoder

---

### 1.2 Import Custom Modules

```python
from metrics import PSNR, SSIM
```

**Triggers execution of metrics.py:**

#### Function: `gaussian(window_size, sigma)` (defined, not executed yet)
```python
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) 
                          for x in range(window_size)])
    return gauss / gauss.sum()
```
**Purpose**: Create 1D Gaussian kernel for SSIM calculation
**Math**: $G(x) = e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

#### Function: `create_window(window_size, channel)` (defined, not executed yet)
```python
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window
```
**Purpose**: Create 2D Gaussian window (11×11) for SSIM
**Process**:
1. Create 1D Gaussian: [11 values]
2. Outer product: [11,1] × [1,11] = [11,11]
3. Expand to channels: [C, 1, 11, 11]

#### Function: `SSIM(img1, img2)` (defined, not executed yet)
```python
def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=5, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=5, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=5, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / 
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
```
**Purpose**: Structural Similarity Index Measurement
**Math Formula**:
$$SSIM(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where:
- $\mu_x, \mu_y$: Local means (computed via Gaussian convolution)
- $\sigma_x, \sigma_y$: Local standard deviations
- $\sigma_{xy}$: Local covariance
- $C_1, C_2$: Stability constants

**Process**:
1. Create 11×11 Gaussian window
2. Convolve with image to get local statistics
3. Compute luminance, contrast, structure components
4. Combine into final SSIM score

#### Function: `PSNR(img1, img2, mask=None)` (defined, not executed yet)
```python
def PSNR(img1, img2, mask=None):
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
```
**Purpose**: Peak Signal-to-Noise Ratio
**Math Formula**:
$$PSNR = 20 \cdot \log_{10}\left(\frac{MAX}{\sqrt{MSE}}\right) = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

Where:
- $MSE = \frac{1}{n}\sum(I_1 - I_2)^2$: Mean Squared Error
- $MAX = 1$: Maximum pixel value (normalized)

**Process**:
1. Compute pixel-wise squared difference
2. Average across all pixels (optionally masked)
3. Convert to decibel scale

---

```python
from dataloader import AlignedDataset, get_train_val_test_filelists
```

**Triggers execution of dataloader.py:**

#### Class: `AlignedDataset(Dataset)` (defined, not instantiated yet)

**Key Attributes**:
```python
self.clip_min = [
    [-25.0, -32.5],  # SAR: VV and VH polarizations
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Optical cloudfree: 13 bands
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Optical cloudy: 13 bands
]
self.clip_max = [
    [0, 0],  # SAR max values
    [10000, 10000, ...],  # Optical max (13 bands)
    [10000, 10000, ...]   # Optical max (13 bands)
]
self.scale = 10000  # Sentinel-2 scaling factor
```

**Purpose**: Sentinel-2 stores pixel values as integers [0-10000] representing reflectance [0-1]

#### Method: `__getitem__(self, index)` (defined, not called yet)
**Full detailed breakdown when called later**

#### Method: `get_sar_image(self, path)` (defined)
```python
def get_sar_image(self, path):
    src = rasterio.open(path, 'r', driver='GTiff')
    image = src.read()  # Returns numpy array (bands, height, width)
    src.close()
    image[np.isnan(image)] = np.nanmean(image)  # Fill NaN with mean
    return image
```
**Purpose**: Load SAR GeoTIFF file and handle missing data
**Input**: File path string
**Output**: Numpy array (2, 256, 256) for VV and VH polarizations
**Process**:
1. Open GeoTIFF using rasterio (wrapper for GDAL)
2. Read all bands into memory
3. Replace NaN values with channel-wise mean (handles artifacts)
4. Close file handle

#### Method: `get_opt_image(self, path)` (defined)
```python
def get_opt_image(self, path):
    src = rasterio.open(path, 'r', driver='GTiff')
    image = src.read()
    src.close()
    image[np.isnan(image)] = np.nanmean(image)
    return image
```
**Purpose**: Load optical (Sentinel-2) GeoTIFF
**Input**: File path
**Output**: Numpy array (13, 256, 256) for 13 Sentinel-2 bands:
- Band 1: Coastal aerosol (443nm)
- Band 2: Blue (490nm)
- Band 3: Green (560nm)
- Band 4: Red (665nm)
- Band 5: Red Edge 1 (705nm)
- Band 6: Red Edge 2 (740nm)
- Band 7: Red Edge 3 (783nm)
- Band 8: NIR (842nm)
- Band 8A: NIR narrow (865nm)
- Band 9: Water vapor (945nm)
- Band 10: Cirrus (1375nm)
- Band 11: SWIR 1 (1610nm)
- Band 12: SWIR 2 (2190nm)

#### Method: `get_normalized_data(self, data_image, data_type)` (defined)
```python
def get_normalized_data(self, data_image, data_type):
    # SAR
    if data_type == 1:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], 
                                         self.clip_min[0][channel], 
                                         self.clip_max[0][channel])
            data_image[channel] -= self.clip_min[0][channel]
            data_image[channel] = self.max_val * (data_image[channel] / 
                                  (self.clip_max[0][channel] - self.clip_min[0][channel]))
    # OPT
    elif data_type == 2 or data_type == 3:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], 
                                         self.clip_min[data_type-1][channel], 
                                         self.clip_max[data_type-1][channel])
        data_image /= self.scale
    return data_image
```
**Purpose**: Normalize different data types to [0, 1] range

**For SAR (type=1)**:
1. Clip to [-25dB, 0dB] (typical SAR backscatter range)
2. Shift by +25 to [0, 25]
3. Divide by 25 to [0, 1]

**For Optical (type=2/3)**:
1. Clip to [0, 10000]
2. Divide by 10000 to [0, 1]

**Why different normalization?**
- SAR: Logarithmic (dB) scale, can be negative
- Optical: Linear reflectance scale, always positive

#### Function: `get_train_val_test_filelists(listpath)` (defined)
```python
def get_train_val_test_filelists(listpath):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)
    
    train_filelist = []
    val_filelist = []
    test_filelist = []
    
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    
    csv_file.close()
    return train_filelist, val_filelist, test_filelist
```
**Purpose**: Parse data.csv to separate train/val/test splits
**Input**: Path to CSV file
**CSV Format**:
```
split_id,s1_folder,s2_cloudfree_folder,s2_cloudy_folder,filename
1,s1,s2_cloudfree,s2_cloudy,train_image.tif
2,s1,s2_cloudfree,s2_cloudy,val_image.tif
3,s1,s2_cloudfree,s2_cloudy,test_image.tif
```
**Output**: Three lists of file metadata

---

```python
from net_CR_RDN import RDN_residual_CR
```

**Triggers execution of net_CR_RDN.py:**

This file defines the entire GLF-CR model architecture. Let's go through every component:

---

#### Function: `pixel_reshuffle(input, upscale_factor)` (defined)
```python
def pixel_reshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    
    input_view = input.contiguous().view(batch_size, channels, 
                                         out_height, upscale_factor, 
                                         out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor
    
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)
```
**Purpose**: **Inverse of PixelShuffle** - downsample spatially, increase channels
**Example**:
```
Input:  (1, 13, 128, 128)  [13-band image, 128×128 pixels]
upscale_factor = 2
→ Reshape to (1, 13, 64, 2, 64, 2)
→ Permute to (1, 13, 2, 2, 64, 64)
→ Reshape to (1, 52, 64, 64)  [52 channels, 64×64 pixels]
```
**Why?**: Trades spatial resolution for channel depth, useful for downsampling before processing

---

#### Class: `DFG(nn.Module)` - Dynamic Filter Generator
```python
class DFG(nn.Module):
    def __init__(self, channels, ks_2d):
        super(DFG, self).__init__()
        ks = 3
        half_channels = channels // 2
        self.fac_warp = nn.Sequential(
            df_conv(channels, half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_conv(half_channels, half_channels * ks_2d ** 2, kernel_size=1))
    
    def forward(self, opt_f, sar_f):
        concat = torch.cat([opt_f, sar_f], 1)
        out = self.fac_warp(concat)
        return out
```
**Purpose**: Generate spatially-varying convolution kernels for dynamic filtering
**Architecture**:
```
Input: OPT features (B, 96, H, W) + SAR features (B, 96, H, W)
   ↓ Concatenate
(B, 192, H, W)
   ↓ 3×3 Conv → (B, 96, H, W)
   ↓ ResBlock (3×3 Conv → LeakyReLU → 3×3 Conv + residual)
   ↓ ResBlock
   ↓ 1×1 Conv → (B, 96×25, H, W)  [25 = 5×5 kernel size]
Output: Dynamic kernels (B, 2400, H, W)
```
**Key Innovation**: Each pixel gets its own unique 5×5 convolutional kernel
**Math**: For pixel $(i,j)$, predict kernel $K_{i,j} \in \mathbb{R}^{C \times 5 \times 5}$

---

#### Class: `Mlp(nn.Module)` - Feed-Forward Network
```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```
**Purpose**: Two-layer MLP used in Transformer blocks
**Architecture**:
```
Input: (B, N, C)
   ↓ Linear: C → C*mlp_ratio (typically C*2)
   ↓ GELU activation
   ↓ Dropout
   ↓ Linear: C*mlp_ratio → C
   ↓ Dropout
Output: (B, N, C)
```
**GELU**: Gaussian Error Linear Unit $GELU(x) = x \cdot \Phi(x)$ where $\Phi$ is CDF of standard normal

---

#### Function: `window_partition(x, window_size)` (defined)
```python
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
```
**Purpose**: Partition image into non-overlapping windows (Swin Transformer style)
**Example**:
```
Input: (1, 128, 128, 96)  [1 image, 128×128 pixels, 96 channels]
window_size = 8
→ Reshape: (1, 16, 8, 16, 8, 96)  [16×16 = 256 windows of 8×8]
→ Permute: (1, 16, 16, 8, 8, 96)
→ Reshape: (256, 8, 8, 96)  [256 separate windows]
```
**Why?**: Limits self-attention to local 8×8 regions, reducing complexity from $O(N^2)$ to $O(w^2 \cdot N)$

---

#### Function: `window_reverse(windows, window_size, H, W)` (defined)
```python
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```
**Purpose**: Inverse of `window_partition` - merge windows back to image
**Example**:
```
Input: (256, 8, 8, 96)  [256 windows]
H=128, W=128, window_size=8
→ Calculate B: 256 / (128*128/8/8) = 1
→ Reshape: (1, 16, 16, 8, 8, 96)
→ Permute: (1, 16, 8, 16, 8, 96)
→ Reshape: (1, 128, 128, 96)  [Back to full image]
```

---

#### Class: `WindowAttention(nn.Module)` - Multi-Head Window Attention

This is one of the most complex components. Let me break it down completely:

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 96 in this model
        self.window_size = window_size  # (8, 8)
        self.num_heads = num_heads  # 8
        head_dim = dim // num_heads  # 96 // 8 = 12 per head
        self.scale = qk_scale or head_dim ** -0.5  # 1/√12 for scaled dot-product
```

**Relative Position Bias Setup**:
```python
        # Parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        # Shape: (15*15, 8) = (225, 8)
        # Why 15? Relative positions range from -7 to +7 (15 values total)
        
        # Compute relative position indices
        coords_h = torch.arange(self.window_size[0])  # [0,1,2,3,4,5,6,7]
        coords_w = torch.arange(self.window_size[1])  # [0,1,2,3,4,5,6,7]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 8, 8)
        coords_flatten = torch.flatten(coords, 1)  # (2, 64)
        
        # Compute pairwise relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Shape: (2, 64, 64) - for each pair of pixels, get (Δh, Δw)
        
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (64, 64, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to [0, 14]
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # Flatten to 1D index
        relative_position_index = relative_coords.sum(-1)  # (64, 64)
        self.register_buffer("relative_position_index", relative_position_index)
```

**Linear Projections**:
```python
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # For optical features
        self.qkv_SAR = nn.Linear(dim, dim * 3, bias=qkv_bias)  # For SAR features
```
**Purpose**: Project features to Query, Key, Value spaces
**Math**: $Q = XW_Q$, $K = XW_K$, $V = XW_V$ where $X \in \mathbb{R}^{N \times C}$

**Attention Fusion**:
```python
        self.attn_fuse_1x1conv = nn.Conv2d(8, 8, kernel_size=1)
```
**Purpose**: Learn to fuse optical and SAR attention maps

**Output Projections**:
```python
        self.proj = nn.Linear(dim, dim)  # Optical
        self.proj_SAR = nn.Linear(dim, dim)  # SAR
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_SAR = nn.Dropout(proj_drop)
```

**Forward Pass**:
```python
    def forward(self, inputs, mask=None):
        [x, x_SAR] = inputs  # Separate optical and SAR
        B_, N, C = x.shape  # (num_windows*B, 64, 96)
```

**Step 1: Compute Q, K, V**:
```python
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Input: (B_, 64, 96)
        # After linear: (B_, 64, 288)  [3 * 96]
        # Reshape: (B_, 64, 3, 8, 12)  [3 for Q/K/V, 8 heads, 12 dims per head]
        # Permute: (3, B_, 8, 64, 12)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B_, 8, 64, 12)
        q_SAR, k_SAR, v_SAR = qkv_SAR[0], qkv_SAR[1], qkv_SAR[2]
```

**Step 2: Scaled Dot-Product Attention**:
```python
        q = q * self.scale  # Multiply by 1/√12
        q_SAR = q_SAR * self.scale
        
        attn = (q @ k.transpose(-2, -1))  # (B_, 8, 64, 64)
        # Math: Attention_raw = Q·K^T / √d_k
        
        attn_SAR = (q_SAR @ k_SAR.transpose(-2, -1))
```

**Step 3: Add Relative Position Bias**:
```python
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(64, 64, -1)  # (64, 64, 8)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (8, 64, 64)
        
        attn = attn + relative_position_bias.unsqueeze(0)  # Broadcast to (B_, 8, 64, 64)
        attn_SAR = attn_SAR + relative_position_bias.unsqueeze(0)
```
**Why?**: Injects 2D spatial inductive bias - pixels closer in space should have stronger correlation

**Step 4: Fuse Optical and SAR Attention**:
```python
        attn_diff_conv = self.attn_fuse_1x1conv(attn_SAR - attn)
        attn_fuse_gate = torch.sigmoid(attn_diff_conv)  # Gating mechanism
        
        attn = attn + (attn_SAR - attn) * attn_fuse_gate
        # Adaptive fusion: learn how much SAR information to incorporate
```
**Math**: $Attn_{fused} = Attn_{opt} + \sigma(Conv(Attn_{SAR} - Attn_{opt})) \odot (Attn_{SAR} - Attn_{opt})$

**Step 5: Softmax and Apply to Values**:
```python
        attn = self.softmax(attn)  # (B_, 8, 64, 64) - normalize attention weights
        attn_SAR = self.softmax(attn_SAR)
        
        attn = self.attn_drop(attn)
        attn_SAR = self.attn_drop_SAR(attn_SAR)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # Math: Output = softmax(Attention) · V
        # (B_, 8, 64, 64) @ (B_, 8, 64, 12) → (B_, 8, 64, 12)
        # Transpose & reshape → (B_, 64, 96)
        
        x_SAR = (attn_SAR @ v_SAR).transpose(1, 2).reshape(B_, N, C)
```

**Step 6: Output Projection**:
```python
        x = self.proj(x)
        x_SAR = self.proj_SAR(x_SAR)
        
        x = self.proj_drop(x)
        x_SAR = self.proj_drop_SAR(x_SAR)
        
        return [x, x_SAR]
```

**Complete Attention Formula**:
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

Where:
- $B$: Relative position bias
- $d_k = 12$: Dimension per head
- Fusion happens at attention map level before softmax

---

#### Class: `RDB_Conv(nn.Module)` - Residual Dense Block Convolution Layer

```python
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize, input_resolution, num_heads, 
                 window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, drop, 
                 attn_drop, drop_path, norm_layer):
        super(RDB_Conv, self).__init__()
        Cin = inChannels  # 96 + c*48 (grows with each layer)
        G = growRate  # 48
        
        # Convolution branches
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])
        self.conv_SAR = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])
```
**Purpose**: Extract local features via convolution, then refine with window attention

**Attention Setup** (detailed earlier):
```python
        self.attn = WindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
```

**MLP Setup**:
```python
        mlp_hidden_dim = int(self.dim * mlp_ratio)  # 48 * 2 = 96
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        self.mlp_SAR = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, 
                          act_layer=act_layer, drop=drop)
```

**Shifted Window Mask** (for layers with shift_size > 0):
```python
        if self.shift_size > 0:
            # Create attention mask for shifted windows
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            
            h_slices = (slice(0, -window_size),
                       slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                       slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
                                   .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
```
**Why Shifted Windows?**: Alternate between regular and shifted window partitioning to enable cross-window connections

**Forward Pass**:
```python
    def forward(self, inputs):
        [input, input_SAR] = inputs
        [x, x_SAR] = inputs  # Same as input, but will be modified
        
        H, W = self.input_resolution  # (64, 64) after downsampling
        
        # Step 1: Convolution
        x_conv = self.conv(x)  # (B, Cin, 64, 64) → (B, 48, 64, 64)
        x_SAR_conv = self.conv_SAR(x_SAR)
        
        # Step 2: Reshape for attention
        x_conv_unfold = x_conv.flatten(2).transpose(1, 2)  # (B, 48, 64, 64) → (B, 4096, 48)
        x_SAR_conv_unfold = x_SAR_conv.flatten(2).transpose(1, 2)
        
        shortcut = x_conv_unfold  # Save for residual connection
        shortcut_SAR = x_SAR_conv_unfold
        
        B, H_W, growRate = x_conv_unfold.shape  # (B, 4096, 48)
        x = x_conv_unfold.view(B, H, W, growRate)  # (B, 64, 64, 48)
        x_SAR = x_SAR_conv_unfold.view(B, H, W, growRate)
        
        # Step 3: Cyclic shift (if shift_size > 0)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Roll by -4 pixels in both H and W directions
            shifted_x_SAR = torch.roll(x_SAR, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_x_SAR = x_SAR
        
        # Step 4: Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        # (B, 64, 64, 48) → (B*64, 8, 8, 48)  [64 windows of 8×8]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, growRate)
        # → (B*64, 64, 48)  [64 pixels per window]
        
        x_SAR_windows = window_partition(shifted_x_SAR, self.window_size)
        x_SAR_windows = x_SAR_windows.view(-1, self.window_size * self.window_size, growRate)
        
        # Step 5: Window-based Multi-Head Self-Attention
        [attn_windows, SAR_attn_windows] = self.attn([x_windows, x_SAR_windows], 
                                                     mask=self.attn_mask)
        # Applies WindowAttention (explained above)
        
        # Step 6: Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, growRate)
        SAR_attn_windows = SAR_attn_windows.view(-1, self.window_size, self.window_size, growRate)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, 64, 64, 48)
        shifted_x_SAR = window_reverse(SAR_attn_windows, self.window_size, H, W)
        
        # Step 7: Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_SAR = torch.roll(shifted_x_SAR, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            x_SAR = shifted_x_SAR
        
        x = x.view(B, H_W, growRate)  # (B, 4096, 48)
        x_SAR = x_SAR.view(B, H_W, growRate)
        
        # Step 8: Residual connection + MLP
        x = shortcut + self.drop_path(x)  # Add residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # Add MLP with residual
        # LayerNorm → MLP → DropPath → Add
        
        x_SAR = shortcut_SAR + self.drop_path_SAR(x_SAR)
        x_SAR = x_SAR + self.drop_path_SAR(self.mlp_SAR(self.norm2_SAR(x_SAR)))
        
        # Step 9: Reshape back to spatial format
        x_unfold = x.transpose(1, 2).view(B, growRate, H, W)  # (B, 48, 64, 64)
        x_SAR_unfold = x_SAR.transpose(1, 2).view(B, growRate, H, W)
        
        # Step 10: Concatenate with input (Dense Connection)
        return [torch.cat((input, x_unfold), 1), torch.cat((input_SAR, x_SAR_unfold), 1)]
        # Cin → Cin + 48 (grows by growRate)
```

**Key Architecture Pattern**:
```
Input (Cin channels)
  ↓ 3×3 Conv → (48 channels)
  ↓ Reshape to sequence
  ↓ Window partition
  ↓ Window Attention (with OPT-SAR fusion)
  ↓ Merge windows
  ↓ Residual connection
  ↓ LayerNorm + MLP + Residual
  ↓ Concatenate with input
Output (Cin + 48 channels)  ← Dense connection!
```

---

#### Class: `RDB(nn.Module)` - Residual Dense Block

```python
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize, input_resolution, 
                 num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop, 
                 attn_drop, drop_path, norm_layer):
        super(RDB, self).__init__()
        G0 = growRate0  # 96 (initial channels)
        G = growRate  # 48 (growth rate)
        C = nConvLayers  # 5 (number of conv layers)
        
        # Create C convolutional layers with alternating window patterns
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(
                inChannels=G0 + c * G,  # Grows: 96, 144, 192, 240, 288
                growRate=G,
                kSize=kSize,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (c % 2 == 0) else window_size // 2,  # Alternate [0,4,0,4,0]
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[c] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)  # (96 + 5*48) → 96
        self.LFF_SAR = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)
    
    def forward(self, inputs):
        [x, x_SAR] = inputs  # (B, 96, 64, 64)
        [x_convs, x_SAR_convs] = self.convs(inputs)  # (B, 336, 64, 64) after all 5 layers
        # 96 + 48*5 = 336 channels
        
        return [self.LFF(x_convs) + x, self.LFF_SAR(x_SAR_convs) + x_SAR]
        # 1×1 Conv to compress back to 96 channels + global residual
```

**Dense Connection Pattern**:
```
Layer 0: 96  →  96+48  = 144 channels
Layer 1: 144 → 144+48  = 192 channels
Layer 2: 192 → 192+48  = 240 channels
Layer 3: 240 → 240+48  = 288 channels
Layer 4: 288 → 288+48  = 336 channels
     ↓ Local Feature Fusion (1×1 Conv)
Output: 96 channels + input (global residual)
```

**Why Dense Connections?**: 
- Each layer receives all previous features
- Encourages feature reuse
- Better gradient flow
- Similar to DenseNet architecture

---

#### Class: `RDN_residual_CR(nn.Module)` - Complete GLF-CR Model

```python
class RDN_residual_CR(nn.Module):
    def __init__(self, input_size):
        super(RDN_residual_CR, self).__init__()
        self.G0 = 96  # Base number of feature channels
        kSize = 3  # Kernel size
        
        # Architecture hyperparameters
        self.D = 6  # Number of RDB blocks
        self.C = 5  # Conv layers per RDB
        self.G = 48  # Growth rate
        
        # Attention hyperparameters
        num_heads = 8
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
```

**Shallow Feature Extraction**:
```python
        # For optical (13 bands → 96 features)
        self.SFENet1 = nn.Conv2d(13 * 4, self.G0, 5, padding=2, stride=1)
        # 13*4 = 52 (after pixel_reshuffle)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=1, stride=1)
        
        # For SAR (2 bands → 96 features)
        self.SFENet1_SAR = nn.Conv2d(2 * 4, self.G0, 5, padding=2, stride=1)
        # 2*4 = 8 (after pixel_reshuffle)
        self.SFENet2_SAR = nn.Conv2d(self.G0, self.G0, kSize, padding=1, stride=1)
```
**Why 13*4 and 2*4?**: After pixel_reshuffle with factor 2, spatial dims halve but channels quadruple

**Stochastic Depth**:
```python
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.D * self.C)]
        # [0.0, 0.0067, 0.0133, ..., 0.2]  [30 values total]
        # Gradually increase drop probability for later layers
```
**Purpose**: Randomly drop entire layers during training for regularization

**RDB Blocks**:
```python
        self.RDBs = nn.ModuleList()
        for i in range(self.D):  # 6 blocks
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C, kSize=kSize,
                    input_resolution=(input_size//2, input_size//2),  # (64, 64)
                    num_heads=num_heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i * self.C:(i + 1) * self.C],  # 5 values per block
                    norm_layer=norm_layer)
            )
```

**Fusion Modules**:
```python
        channels = self.G0  # 96
        ks_2d = 5  # Kernel size for dynamic filtering
        
        # Dynamic Filter Generator (one per RDB block)
        self.DF = nn.ModuleList()
        for i in range(self.D):
            self.DF.append(DFG(channels*2, ks_2d))  # Input: 192 channels
        
        # Dynamic Filter Refiner (custom CUDA kernel)
        self.DFR = nn.ModuleList()
        for i in range(self.D):
            self.DFR.append(KernelConv2D.KernelConv2D(ks_2d))
        
        # Gating mechanisms for fusion
        self.sar_fuse_1x1conv = nn.ModuleList()  # SAR → OPT
        for i in range(self.D):
            self.sar_fuse_1x1conv.append(nn.Conv2d(channels, channels, kernel_size=1))
        
        self.opt_distribute_1x1conv = nn.ModuleList()  # OPT → SAR
        for i in range(self.D):
            self.opt_distribute_1x1conv.append(nn.Conv2d(channels, channels, kernel_size=1))
```

**Global Feature Fusion**:
```python
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),  # 576 → 96
            nn.Conv2d(self.G0, self.G0, kSize, padding=1, stride=1)
        ])
```
**Purpose**: Fuse outputs from all 6 RDB blocks

**Upsampling**:
```python
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=1, stride=1),
            nn.PixelShuffle(2),  # 256 channels → 64 channels, 2× spatial size
            nn.Conv2d(64, 13, kSize, padding=1, stride=1)  # Back to 13 bands
        ])
```
**PixelShuffle**: Rearranges (C×r², H, W) → (C, H×r, W×r) for efficient upsampling

---

**Forward Pass** (THE MAIN EVENT):

```python
    def forward(self, cloudy_data, SAR):
        # cloudy_data: (1, 13, 128, 128)
        # SAR: (1, 2, 128, 128)
        
        # Step 1: Pixel Reshuffle (trade spatial for channel)
        B_shuffle = pixel_reshuffle(cloudy_data, 2)
        # (1, 13, 128, 128) → (1, 52, 64, 64)
        
        f__1 = self.SFENet1(B_shuffle)  # 5×5 Conv: (1, 52, 64, 64) → (1, 96, 64, 64)
        x = self.SFENet2(f__1)  # 3×3 Conv: (1, 96, 64, 64) → (1, 96, 64, 64)
        
        B_shuffle_SAR = pixel_reshuffle(SAR, 2)
        # (1, 2, 128, 128) → (1, 8, 64, 64)
        
        f__1__SAR = self.SFENet1_SAR(B_shuffle_SAR)  # (1, 8, 64, 64) → (1, 96, 64, 64)
        x_SAR = self.SFENet2_SAR(f__1__SAR)  # (1, 96, 64, 64) → (1, 96, 64, 64)
        
        # Step 2: RDB Blocks with Fusion
        RDBs_out = []
        for i in range(self.D):  # 6 iterations
            # RDB processing (5 conv layers with window attention)
            [x, x_SAR] = self.RDBs[i]([x, x_SAR])
            # Input: (1, 96, 64, 64) each
            # Output: (1, 96, 64, 64) each (local feature fusion brings back to 96)
            
            # Dynamic filtering fusion
            x, x_SAR = self.fuse(x, x_SAR, i)
            
            RDBs_out.append(x)  # Save optical features for GFF
        
        # Step 3: Global Feature Fusion
        x = self.GFF(torch.cat(RDBs_out, 1))  # Concat 6×96 → 576, then 1×1 Conv → 96
        x += f__1  # Add shallow features (global residual)
        
        # Step 4: Upsampling
        pred_CloudFree_data = self.UPNet(x) + cloudy_data
        # (1, 96, 64, 64) → (1, 256, 64, 64) → (1, 64, 128, 128) → (1, 13, 128, 128)
        # Add input as final residual
        
        return pred_CloudFree_data  # (1, 13, 128, 128)
```

**Fusion Method** (called 6 times, once per RDB):
```python
    def fuse(self, OPT, SAR, i):
        OPT_m = OPT  # (1, 96, 64, 64)
        SAR_m = SAR  # (1, 96, 64, 64)
        
        # Step 1: Generate dynamic kernels from both modalities
        kernel_sar = self.DF[i](OPT_m, SAR_m)
        # Concat → (1, 192, 64, 64)
        # DFG network → (1, 96*25, 64, 64)  [2400 channels = 96 features × 25 kernel weights]
        
        # Step 2: Apply dynamic convolution to SAR
        SAR_m = self.DFR[i](SAR_m, kernel_sar)
        # Custom CUDA kernel: Each pixel gets unique 5×5 conv kernel
        # (1, 96, 64, 64) + (1, 2400, 64, 64) → (1, 96, 64, 64)
        
        # Step 3: Compute gating signal for SAR→OPT fusion
        sar_s = self.sar_fuse_1x1conv[i](SAR_m - OPT_m)  # 1×1 Conv on difference
        sar_fuse_gate = torch.sigmoid(sar_s)  # Gating signal ∈ [0, 1]
        
        # Step 4: Update optical features
        new_OPT = OPT + (SAR_m - OPT_m) * sar_fuse_gate
        # Adaptively add SAR information where gate is high
        
        new_OPT_m = new_OPT
        
        # Step 5: Compute gating signal for OPT→SAR distribution
        opt_s = self.opt_distribute_1x1conv[i](new_OPT_m - SAR_m)
        opt_distribute_gate = torch.sigmoid(opt_s)
        
        # Step 6: Update SAR features
        new_SAR = SAR + (new_OPT_m - SAR_m) * opt_distribute_gate
        # Bidirectional fusion!
        
        return new_OPT, new_SAR
```

**Fusion Math**:
$$OPT_{new} = OPT + \sigma(Conv_{1×1}(SAR' - OPT)) \odot (SAR' - OPT)$$
$$SAR_{new} = SAR + \sigma(Conv_{1×1}(OPT_{new} - SAR')) \odot (OPT_{new} - SAR')$$

Where:
- $SAR' = DynamicConv(SAR, Kernels)$: SAR warped by predicted kernels
- $\sigma$: Sigmoid gating
- $\odot$: Element-wise multiplication

**Why Bidirectional?**: 
- SAR provides structure under clouds
- Optical provides spectral information
- Adaptive fusion learns what to take from each modality

---

## COMPLETE MODEL ARCHITECTURE SUMMARY

```
Input: Cloudy Optical (1,13,128,128) + SAR (1,2,128,128)
   ↓
[PIXEL RESHUFFLE] → (1,52,64,64) + (1,8,64,64)
   ↓
[SHALLOW FEATURE EXTRACTION]
   5×5 Conv → (1,96,64,64) for both
   3×3 Conv → (1,96,64,64) for both
   ↓
[RDB BLOCK 1] ←────────┐
   5× (Conv3×3 + WindowAttn + MLP)  │
   Dense connections: 96→144→192→240→288→336  │
   Local Feat Fusion: 336→96  │
   [DYNAMIC FUSION] SAR↔OPT  │
   Output: (1,96,64,64) each  │
   ↓  │
[RDB BLOCK 2-6] ──────────┤ (repeat 5 more times)
   ↓  │
[GLOBAL FEATURE FUSION]  ← Concatenate all 6 RDB outputs
   Concat: 6×96 = 576 channels
   1×1 Conv: 576→96
   3×3 Conv: 96→96
   + Add shallow features
   ↓
[UPSAMPLING]
   3×3 Conv: 96→256
   PixelShuffle(2×): (1,64,128,128)
   3×3 Conv: 64→13
   + Add input (global residual)
   ↓
Output: Cloud-free Optical (1,13,128,128)
```

**Total Parameter Count**: ~5-10 million parameters

**Key Innovations**:
1. **Dual-stream processing**: Separate encoders for optical and SAR
2. **Window attention**: Efficient local self-attention (8×8 windows)
3. **Dynamic filtering**: Spatially-varying kernels via custom CUDA
4. **Bidirectional fusion**: SAR↔OPT information exchange
5. **Dense connections**: Feature reuse within RDB blocks
6. **Multi-scale residuals**: Local (RDB), global (GFF), input (final)

---

Now let's continue with the actual execution when you run the command...

