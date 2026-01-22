# GLF-CR Enhancement Implementation Plan
## Progressive Transformer Integration - Hybrid Model Development

**Base Paper**: GLF-CR (Global-Local Fusion for Cloud Removal)  
**Goal**: Enhance GLF-CR with transformer components while maintaining its core architecture  
**Timeline**: 8-10 weeks  
**Date**: November 19, 2025

---

## 🎯 Project Objectives

1. **Preserve GLF-CR Identity**: Keep the core RDN + dynamic fusion architecture
2. **Progressive Enhancement**: Add transformers incrementally, validate each addition
3. **Hybrid Architecture**: CNN (GLF-CR base) + Transformer enhancements
4. **Measurable Improvements**: Track PSNR/SSIM gains at each step
5. **Reproducibility**: Document and version each enhancement

---

## 📋 Implementation Phases

### **PHASE 0: Preparation & Baseline** (Week 1)
**Status**: 🔴 Not Started  
**Duration**: 5-7 days

#### Objectives:
- Establish reproducible baseline
- Set up proper experiment tracking
- Prepare data pipeline
- Create modular code structure

#### Tasks:

**Task 0.1: Environment Setup**
```bash
# Create dedicated branch for enhancements
git checkout -b glf-cr-transformer-enhancements

# Document current environment
pip freeze > requirements_baseline.txt

# Install additional dependencies
pip install tensorboard wandb timm einops
```

**Task 0.2: Baseline Evaluation**
- [ ] Run current GLF-CR model on validation set
- [ ] Record baseline metrics in `results/baseline_metrics.json`:
  ```json
  {
    "model": "GLF-CR (baseline)",
    "date": "2025-11-19",
    "psnr_avg": 0.0,
    "ssim_avg": 0.0,
    "psnr_std": 0.0,
    "ssim_std": 0.0,
    "num_samples": 0,
    "inference_time_ms": 0.0
  }
  ```
- [ ] Document model size and FLOPs
- [ ] Set up TensorBoard/WandB logging

**Task 0.3: Code Restructuring**
```
codes/
├── model_CR_net.py              # Keep as-is (base model wrapper)
├── net_CR_RDN.py                # Keep as-is (core GLF-CR architecture)
├── submodules.py                # Keep as-is
├── feature_detectors.py         # Keep as-is
├── dataloader.py                # Keep as-is
├── metrics.py                   # Keep as-is
│
├── enhancements/                # NEW: All transformer enhancements
│   ├── __init__.py
│   ├── cross_modal_attn.py     # Phase 1
│   ├── perceptual_loss.py      # Phase 1
│   ├── texture_branch.py       # Phase 2
│   ├── global_local_attn.py    # Phase 3
│   └── utils.py
│
├── models/                      # NEW: Enhanced model versions
│   ├── __init__.py
│   ├── model_CR_v1.py          # GLF-CR + Cross-Modal Attn
│   ├── model_CR_v2.py          # v1 + Perceptual Loss
│   ├── model_CR_v3.py          # v2 + Texture Branch
│   └── model_CR_final.py       # All enhancements
│
├── train_enhanced.py            # NEW: Training with experiment tracking
├── test_enhanced.py             # NEW: Testing with detailed metrics
└── config.py                    # NEW: Configuration management
```

**Task 0.4: Configuration System**
Create `codes/config.py`:
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Base GLF-CR settings
    crop_size: int = 128
    num_RDB: int = 6
    num_conv_layers: int = 5
    growth_rate: int = 48
    
    # Enhancement flags
    use_cross_modal_attn: bool = False
    use_perceptual_loss: bool = False
    use_texture_branch: bool = False
    use_global_attn: bool = False
    
    # Training settings
    lr: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 100
    
    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.0
    ssim_weight: float = 0.0
    texture_weight: float = 0.0
```

**Deliverables**:
- ✅ Baseline metrics documented
- ✅ Code restructured with modular design
- ✅ Experiment tracking set up
- ✅ Git branch created and initial commit

---

### **PHASE 1: Cross-Modal Attention** (Week 2-3)
**Status**: 🔴 Not Started  
**Duration**: 10-12 days  
**Expected Gain**: +1.5-2.0 dB PSNR, +0.03-0.04 SSIM

#### Rationale:
The current GLF-CR processes optical and SAR features separately, then fuses them with dynamic filtering. Adding **explicit cross-modal attention** will allow the model to learn which SAR features are most relevant for reconstructing specific optical regions.

#### Implementation Strategy:

**Task 1.1: Create Cross-Modal Attention Module**

Create `codes/enhancements/cross_modal_attn.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-attention between optical and SAR features.
    Maintains GLF-CR's dual-stream architecture while adding explicit attention.
    """
    def __init__(self, dim=96, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Optical queries, SAR keys/values
        self.q_optical = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_sar = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # SAR queries, optical keys/values
        self.q_sar = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_optical = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Projection and fusion
        self.proj_optical = nn.Linear(dim, dim)
        self.proj_sar = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gating mechanism (inspired by GLF-CR's fusion gates)
        self.gate_optical = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_sar = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: [B, N, C] optical features
            sar_feat: [B, N, C] SAR features
        Returns:
            enhanced_optical: [B, N, C]
            enhanced_sar: [B, N, C]
        """
        B, N, C = optical_feat.shape
        
        # Optical attends to SAR
        q_opt = self.q_optical(optical_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_sar = self.kv_sar(sar_feat).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_sar, v_sar = kv_sar[0], kv_sar[1]
        
        attn_opt = (q_opt @ k_sar.transpose(-2, -1)) * self.scale
        attn_opt = attn_opt.softmax(dim=-1)
        attn_opt = self.attn_drop(attn_opt)
        
        optical_attended = (attn_opt @ v_sar).transpose(1, 2).reshape(B, N, C)
        optical_attended = self.proj_drop(self.proj_optical(optical_attended))
        
        # SAR attends to optical
        q_sar = self.q_sar(sar_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_opt = self.kv_optical(optical_feat).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_opt, v_opt = kv_opt[0], kv_opt[1]
        
        attn_sar = (q_sar @ k_opt.transpose(-2, -1)) * self.scale
        attn_sar = attn_sar.softmax(dim=-1)
        attn_sar = self.attn_drop(attn_sar)
        
        sar_attended = (attn_sar @ v_opt).transpose(1, 2).reshape(B, N, C)
        sar_attended = self.proj_drop(self.proj_sar(sar_attended))
        
        # Gated fusion (inspired by GLF-CR's gate mechanism)
        gate_opt = self.gate_optical(optical_attended)
        gate_sar = self.gate_sar(sar_attended)
        
        enhanced_optical = optical_feat + gate_opt * optical_attended
        enhanced_sar = sar_feat + gate_sar * sar_attended
        
        return enhanced_optical, enhanced_sar
```

**Task 1.2: Integrate into GLF-CR Architecture**

Modify `codes/net_CR_RDN.py` - Add cross-attention AFTER each RDB block:
```python
# In RDB class __init__, add:
self.cross_modal_attn = CrossModalAttention(dim=G0) if use_cross_attn else None

# In RDB forward method, add:
def forward(self, inputs):
    [x, x_SAR] = inputs
    [x_convs, x_SAR_convs] = self.convs(inputs)
    
    x_out = self.LFF(x_convs) + x
    x_SAR_out = self.LFF_SAR(x_SAR_convs) + x_SAR
    
    # NEW: Cross-modal attention
    if self.cross_modal_attn is not None:
        B, C, H, W = x_out.shape
        x_flat = x_out.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_sar_flat = x_SAR_out.flatten(2).transpose(1, 2)
        
        x_flat, x_sar_flat = self.cross_modal_attn(x_flat, x_sar_flat)
        
        x_out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        x_SAR_out = x_sar_flat.transpose(1, 2).reshape(B, C, H, W)
    
    return [x_out, x_SAR_out]
```

**Task 1.3: Create v1 Model**

Create `codes/models/model_CR_v1.py`:
```python
from net_CR_RDN import RDN_residual_CR
import torch.nn as nn

class RDN_residual_CR_v1(RDN_residual_CR):
    """GLF-CR v1: Base model + Cross-Modal Attention"""
    def __init__(self, input_size, use_cross_attn=True):
        super().__init__(input_size)
        
        # Enable cross-attention in RDB blocks
        for i, rdb in enumerate(self.RDBs):
            # Add cross-attention to RDB blocks 2, 3, 4 (middle layers)
            if i in [2, 3, 4] and use_cross_attn:
                from enhancements.cross_modal_attn import CrossModalAttention
                rdb.cross_modal_attn = CrossModalAttention(dim=self.G0)
```

**Task 1.4: Training & Validation**
```bash
# Train v1 model
python codes/train_enhanced.py \
    --model_version v1 \
    --use_cross_modal_attn \
    --epochs 100 \
    --save_dir results/v1_cross_attn

# Evaluate
python codes/test_enhanced.py \
    --model_version v1 \
    --checkpoint results/v1_cross_attn/best_model.pth \
    --save_results results/v1_cross_attn/test_results.json
```

**Task 1.5: Ablation Study**
Test cross-attention at different positions:
- [ ] RDB blocks [2, 3, 4] (default)
- [ ] All RDB blocks [0-5]
- [ ] Only final block [5]
- [ ] Compare PSNR/SSIM/inference time

**Success Criteria**:
- ✅ PSNR improvement: +1.5 dB minimum
- ✅ SSIM improvement: +0.03 minimum
- ✅ Inference time increase: <30%
- ✅ Visual quality improved (especially in cloud regions)

**Deliverables**:
- ✅ Cross-modal attention module implemented
- ✅ v1 model trained and validated
- ✅ Results documented in `results/v1_cross_attn/report.md`
- ✅ Code committed to git with tag `v1.0-cross-attn`

---

### **PHASE 2: Perceptual Loss Integration** (Week 3-4)
**Status**: 🔴 Not Started  
**Duration**: 7-10 days  
**Expected Gain**: +1.0-1.5 dB PSNR, +0.03-0.05 SSIM

#### Rationale:
Current GLF-CR uses only L1 loss, which can lead to blurry reconstructions. Adding perceptual loss using pre-trained vision features will improve perceptual quality and texture preservation.

#### Implementation Strategy:

**Task 2.1: Create Perceptual Loss Module**

Create `codes/enhancements/perceptual_loss.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Lightweight alternative to ViT for faster training.
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        
        # Use VGG16 pre-trained on ImageNet
        vgg = models.vgg16(pretrained=True).features
        
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }
        
        self.layers = layers
        self.model = nn.ModuleDict()
        
        # Extract features at specified layers
        for name, layer in vgg.named_children():
            if name in self.layer_name_mapping:
                self.model[self.layer_name_mapping[name]] = nn.Sequential(
                    *list(vgg.children())[:int(name)+1]
                )
        
        # Freeze VGG
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Normalize for ImageNet pre-training
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize_input(self, x):
        """Normalize for VGG (expects RGB ImageNet normalized)"""
        # Assume x is 13-channel Sentinel-2, use first 3 bands as RGB proxy
        rgb = x[:, [3, 2, 1], :, :]  # Red, Green, Blue bands
        rgb = (rgb - self.mean) / self.std
        return rgb
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 13, H, W] predicted image
            target: [B, 13, H, W] target image
        Returns:
            perceptual_loss: scalar
        """
        # Convert to RGB
        pred_rgb = self.normalize_input(pred)
        target_rgb = self.normalize_input(target)
        
        loss = 0.0
        for layer_name in self.layers:
            pred_feat = self.model[layer_name](pred_rgb)
            target_feat = self.model[layer_name](target_rgb)
            loss += F.mse_loss(pred_feat, target_feat)
            
        return loss / len(self.layers)


class SSIMLoss(nn.Module):
    """SSIM loss for better structural similarity"""
    def __init__(self, window_size=11):
        super().__init__()
        from metrics import SSIM
        self.ssim = SSIM
        
    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)
```

**Task 2.2: Enhanced Loss Function**

Create `codes/enhancements/losses.py`:
```python
import torch.nn as nn
from enhancements.perceptual_loss import PerceptualLoss, SSIMLoss

class EnhancedLoss(nn.Module):
    """
    Enhanced loss for GLF-CR:
    - L1 loss (base, from original paper)
    - Perceptual loss (VGG features)
    - SSIM loss (structural similarity)
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, ssim_weight=0.1):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
        
    def forward(self, pred, target):
        loss_dict = {}
        
        # Base L1 loss (always computed)
        l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = l1
        total_loss = self.l1_weight * l1
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
            loss_dict['perceptual'] = perceptual
            total_loss += self.perceptual_weight * perceptual
            
        # SSIM loss
        if self.ssim_loss is not None:
            ssim = self.ssim_loss(pred, target)
            loss_dict['ssim'] = ssim
            total_loss += self.ssim_weight * ssim
            
        loss_dict['total'] = total_loss
        return total_loss, loss_dict
```

**Task 2.3: Update Training Script**

Modify `codes/train_enhanced.py`:
```python
# Replace simple L1 loss with enhanced loss
from enhancements.losses import EnhancedLoss

# In training loop:
criterion = EnhancedLoss(
    l1_weight=1.0,
    perceptual_weight=0.1,  # Start small, tune if needed
    ssim_weight=0.1
)

# In optimization step:
pred = model(cloudy_data, SAR_data)
loss, loss_dict = criterion(pred, cloudfree_data)

# Log all loss components
for loss_name, loss_value in loss_dict.items():
    logger.log({f'train/{loss_name}': loss_value})
```

**Task 2.4: Training & Validation**
```bash
# Train v2 model (v1 + perceptual loss)
python codes/train_enhanced.py \
    --model_version v1 \
    --use_cross_modal_attn \
    --use_perceptual_loss \
    --perceptual_weight 0.1 \
    --ssim_weight 0.1 \
    --epochs 100 \
    --save_dir results/v2_perceptual

# Evaluate
python codes/test_enhanced.py \
    --model_version v2 \
    --checkpoint results/v2_perceptual/best_model.pth \
    --save_results results/v2_perceptual/test_results.json
```

**Task 2.5: Loss Weight Tuning**
- [ ] Test perceptual_weight: [0.05, 0.1, 0.2]
- [ ] Test ssim_weight: [0.05, 0.1, 0.2]
- [ ] Find optimal balance for PSNR vs perceptual quality

**Success Criteria**:
- ✅ PSNR improvement: +1.0 dB over v1
- ✅ SSIM improvement: +0.03 over v1
- ✅ Perceptual quality improved (texture/edges)
- ✅ No training instability

**Deliverables**:
- ✅ Perceptual loss implemented
- ✅ v2 model trained with enhanced loss
- ✅ Loss weight ablation study completed
- ✅ Results documented and committed with tag `v2.0-perceptual`

---

### **PHASE 3: Texture Enhancement Branch** (Week 5-6)
**Status**: 🔴 Not Started  
**Duration**: 10-12 days  
**Expected Gain**: +0.5-1.0 dB PSNR, +0.02-0.03 SSIM

#### Rationale:
Cloud removal often loses high-frequency details. Adding a dedicated texture enhancement branch will help preserve fine details and edges.

#### Implementation Strategy:

**Task 3.1: Create Texture Branch Module**

Create `codes/enhancements/texture_branch.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFrequencyExtractor(nn.Module):
    """Extract high-frequency components using high-pass filtering"""
    def __init__(self, channels):
        super().__init__()
        
        # Learnable high-pass filter
        self.high_pass = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
        # Low-pass (for residual)
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        low = self.low_pass(x)
        high = x - low
        high_enhanced = self.high_pass(high)
        return high_enhanced, low


class TextureEnhancementBranch(nn.Module):
    """
    Dedicated branch for texture and detail enhancement.
    Operates in parallel to main GLF-CR pipeline.
    """
    def __init__(self, channels=96, num_blocks=3):
        super().__init__()
        
        # High-frequency extraction
        self.hf_extractor = HighFrequencyExtractor(channels)
        
        # Texture refinement blocks
        self.texture_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ) for _ in range(num_blocks)
        ])
        
        # Edge-aware attention
        self.edge_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Fusion with main branch
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
    def forward(self, main_feat):
        """
        Args:
            main_feat: [B, C, H, W] features from main GLF-CR pipeline
        Returns:
            enhanced_feat: [B, C, H, W] texture-enhanced features
            edge_map: [B, 1, H, W] edge attention map
        """
        # Extract high-frequency components
        high_freq, low_freq = self.hf_extractor(main_feat)
        
        # Refine textures
        texture_feat = high_freq
        for block in self.texture_blocks:
            texture_feat = texture_feat + block(texture_feat)
        
        # Compute edge attention map
        edge_map = self.edge_attention(texture_feat)
        
        # Apply edge-aware weighting
        texture_feat = texture_feat * edge_map
        
        # Fuse with main features
        fused = self.fusion(torch.cat([main_feat, texture_feat], dim=1))
        
        return fused, edge_map


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss that emphasizes high-frequency regions.
    Uses edge map to weight reconstruction loss.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, edge_map):
        """
        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image
            edge_map: [B, 1, H, W] edge attention map
        """
        # Standard L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # Weight by edge map (emphasize edges)
        weighted_loss = l1_loss * (1.0 + 2.0 * edge_map)
        
        return weighted_loss.mean()
```

**Task 3.2: Integrate Texture Branch**

Modify `codes/net_CR_RDN.py` in the `RDN_residual_CR` class:
```python
class RDN_residual_CR_v3(RDN_residual_CR):
    """GLF-CR v3: v2 + Texture Enhancement Branch"""
    def __init__(self, input_size, use_texture_branch=True):
        super().__init__(input_size)
        
        if use_texture_branch:
            from enhancements.texture_branch import TextureEnhancementBranch
            # Add texture branch after GFF (Global Feature Fusion)
            self.texture_branch = TextureEnhancementBranch(channels=self.G0)
        else:
            self.texture_branch = None
    
    def forward(self, cloudy_data, SAR):
        # Original GLF-CR forward pass
        B_shuffle = pixel_reshuffle(cloudy_data, 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        
        B_shuffle_SAR = pixel_reshuffle(SAR, 2)
        f__1__SAR = self.SFENet1_SAR(B_shuffle_SAR)
        x_SAR = self.SFENet2_SAR(f__1__SAR)
        
        RDBs_out = []
        for i in range(self.D):
            [x, x_SAR] = self.RDBs[i]([x, x_SAR])
            x, x_SAR = self.fuse(x, x_SAR, i)
            RDBs_out.append(x)
        
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        
        # NEW: Texture enhancement
        edge_map = None
        if self.texture_branch is not None:
            x, edge_map = self.texture_branch(x)
        
        pred_CloudFree_data = self.UPNet(x) + cloudy_data
        
        return pred_CloudFree_data, edge_map
```

**Task 3.3: Update Loss Function**

Modify `codes/enhancements/losses.py`:
```python
class EnhancedLossV3(nn.Module):
    """Enhanced loss with texture/edge awareness"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, 
                 ssim_weight=0.1, edge_weight=0.2):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeAwareLoss()
        
    def forward(self, pred, target, edge_map=None):
        loss_dict = {}
        
        # Base losses
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.perceptual_weight * perceptual +
                     self.ssim_weight * ssim)
        
        loss_dict.update({'l1': l1, 'perceptual': perceptual, 'ssim': ssim})
        
        # Edge-aware loss if edge map provided
        if edge_map is not None and self.edge_weight > 0:
            edge = self.edge_loss(pred, target, edge_map)
            total_loss += self.edge_weight * edge
            loss_dict['edge'] = edge
            
        loss_dict['total'] = total_loss
        return total_loss, loss_dict
```

**Task 3.4: Training & Validation**
```bash
# Train v3 model
python codes/train_enhanced.py \
    --model_version v3 \
    --use_cross_modal_attn \
    --use_perceptual_loss \
    --use_texture_branch \
    --edge_weight 0.2 \
    --epochs 100 \
    --save_dir results/v3_texture

# Evaluate
python codes/test_enhanced.py \
    --model_version v3 \
    --checkpoint results/v3_texture/best_model.pth \
    --save_results results/v3_texture/test_results.json \
    --visualize_edge_maps
```

**Task 3.5: Visual Quality Assessment**
- [ ] Generate edge maps for validation images
- [ ] Compare texture preservation vs baseline
- [ ] Measure improvement on high-frequency regions
- [ ] User study (optional): perceptual quality rating

**Success Criteria**:
- ✅ PSNR improvement: +0.5 dB over v2
- ✅ SSIM improvement: +0.02 over v2
- ✅ Better edge/texture preservation (visual inspection)
- ✅ Edge maps meaningful and useful

**Deliverables**:
- ✅ Texture enhancement branch implemented
- ✅ v3 model trained and validated
- ✅ Edge-aware loss integrated
- ✅ Visual comparison results documented
- ✅ Code committed with tag `v3.0-texture`

---

### **PHASE 4: Global-Local Attention** (Week 7-8)
**Status**: 🔴 Not Started  
**Duration**: 10-14 days  
**Expected Gain**: +1.5-2.5 dB PSNR, +0.04-0.06 SSIM

#### Rationale:
Current window attention (8x8) limits long-range dependencies. Adding global attention path will capture broader context for better cloud removal.

#### Implementation Strategy:

**Task 4.1: Create Global-Local Attention Module**

Create `codes/enhancements/global_local_attn.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientGlobalAttention(nn.Module):
    """
    Efficient global attention using downsampling.
    Maintains computational efficiency while capturing long-range dependencies.
    """
    def __init__(self, dim, num_heads=8, downsample_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.downsample_ratio = downsample_ratio
        
        # Downsample for efficiency
        self.downsample = nn.Conv2d(dim, dim, 
                                    kernel_size=downsample_ratio, 
                                    stride=downsample_ratio)
        
        # Global attention
        self.global_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Upsample back
        self.upsample = nn.ConvTranspose2d(dim, dim,
                                          kernel_size=downsample_ratio,
                                          stride=downsample_ratio)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            global_feat: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Downsample
        x_down = self.downsample(x)  # [B, C, H//r, W//r]
        _, _, H_d, W_d = x_down.shape
        
        # Flatten for attention
        x_flat = x_down.flatten(2).transpose(1, 2)  # [B, H_d*W_d, C]
        x_flat = self.norm(x_flat)
        
        # Global attention
        global_feat, _ = self.global_attn(x_flat, x_flat, x_flat)
        
        # Reshape and upsample
        global_feat = global_feat.transpose(1, 2).reshape(B, C, H_d, W_d)
        global_feat = self.upsample(global_feat)  # [B, C, H, W]
        
        return global_feat


class GlobalLocalFusion(nn.Module):
    """
    Fuse local window attention (existing) with global attention (new).
    Maintains GLF-CR's local processing while adding global context.
    """
    def __init__(self, dim, num_heads=8, window_size=8, downsample_ratio=4):
        super().__init__()
        
        # Local path (use existing WindowAttention from GLF-CR)
        from net_CR_RDN import WindowAttention
        self.local_attn = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads
        )
        
        # Global path (new)
        self.global_attn = EfficientGlobalAttention(
            dim=dim,
            num_heads=num_heads,
            downsample_ratio=downsample_ratio
        )
        
        # Adaptive fusion
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LayerNorm([dim, 1, 1]),  # Will broadcast
            nn.Conv2d(dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [x, x_SAR] where x: [B, C, H, W]
        Returns:
            [enhanced_x, enhanced_x_SAR]
        """
        [x, x_SAR] = inputs
        B, C, H, W = x.shape
        
        # Process optical features
        # Local attention (existing)
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        local_feat, _ = self.local_attn([x_flat, x_flat])  # Reuse existing window attn
        local_feat = local_feat.transpose(1, 2).reshape(B, C, H, W)
        
        # Global attention (new)
        global_feat = self.global_attn(x)
        
        # Adaptive fusion
        fusion_input = torch.cat([local_feat, global_feat], dim=1)
        weights = self.fusion_weight(fusion_input)  # [B, 2, H, W]
        
        fused = weights[:, 0:1] * local_feat + weights[:, 1:2] * global_feat
        enhanced_x = x + self.proj(fused)
        
        # Similarly for SAR (optional: can use same global context)
        x_sar_flat = x_SAR.flatten(2).transpose(1, 2)
        local_feat_sar, _ = self.local_attn([x_sar_flat, x_sar_flat])
        local_feat_sar = local_feat_sar.transpose(1, 2).reshape(B, C, H, W)
        global_feat_sar = self.global_attn(x_SAR)
        
        fusion_input_sar = torch.cat([local_feat_sar, global_feat_sar], dim=1)
        weights_sar = self.fusion_weight(fusion_input_sar)
        
        fused_sar = weights_sar[:, 0:1] * local_feat_sar + weights_sar[:, 1:2] * global_feat_sar
        enhanced_x_SAR = x_SAR + self.proj(fused_sar)
        
        return [enhanced_x, enhanced_x_SAR]
```

**Task 4.2: Integrate Global-Local Attention**

Modify `codes/net_CR_RDN.py`:
```python
class RDN_residual_CR_v4(RDN_residual_CR_v3):
    """GLF-CR v4 (FINAL): v3 + Global-Local Attention"""
    def __init__(self, input_size, use_global_local=True):
        super().__init__(input_size, use_texture_branch=True)
        
        if use_global_local:
            from enhancements.global_local_attn import GlobalLocalFusion
            
            # Replace attention in specific RDB blocks with global-local
            # Apply to blocks 3, 4, 5 (deeper layers benefit more from global context)
            for i in [3, 4, 5]:
                self.RDBs[i].global_local_attn = GlobalLocalFusion(
                    dim=self.G0,
                    num_heads=8,
                    window_size=8,
                    downsample_ratio=4
                )
        
    # Update forward pass to use global-local attention where added
```

**Task 4.3: Computational Optimization**
```python
# Use gradient checkpointing for memory efficiency
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    if self.training:
        return checkpoint(self._forward_impl, x)
    else:
        return self._forward_impl(x)
```

**Task 4.4: Training & Validation**
```bash
# Train v4 (FINAL) model
python codes/train_enhanced.py \
    --model_version v4 \
    --use_cross_modal_attn \
    --use_perceptual_loss \
    --use_texture_branch \
    --use_global_local \
    --gradient_checkpointing \
    --epochs 150 \
    --save_dir results/v4_final

# Evaluate
python codes/test_enhanced.py \
    --model_version v4 \
    --checkpoint results/v4_final/best_model.pth \
    --save_results results/v4_final/test_results.json \
    --compare_all_versions
```

**Task 4.5: Efficiency Analysis**
- [ ] Measure FLOPs for each version (v0-v4)
- [ ] Measure inference time on different image sizes
- [ ] Memory consumption analysis
- [ ] Throughput (images/sec) comparison

**Success Criteria**:
- ✅ PSNR improvement: +1.5 dB over v3
- ✅ SSIM improvement: +0.04 over v3
- ✅ Inference time increase: <2.5x vs baseline
- ✅ Better global context modeling (large cloud regions)

**Deliverables**:
- ✅ Global-local attention implemented
- ✅ Final v4 model trained and validated
- ✅ Efficiency analysis completed
- ✅ Code committed with tag `v4.0-final`

---

### **PHASE 5: Final Evaluation & Documentation** (Week 9-10)
**Status**: 🔴 Not Started  
**Duration**: 7-10 days

#### Objectives:
- Comprehensive evaluation of all versions
- Ablation study
- Documentation and paper preparation
- Model deployment

#### Tasks:

**Task 5.1: Comprehensive Evaluation**
```bash
# Run all models on full test set
python codes/evaluate_all_versions.py \
    --test_data_dir data/ \
    --models_dir results/ \
    --output_dir results/final_evaluation/
```

Create comparison table:
```
| Model Version | PSNR (dB) | SSIM | MAE | Params (M) | FLOPs (G) | Time (ms) |
|---------------|-----------|------|-----|------------|-----------|-----------|
| GLF-CR (base) |   X.XX    | 0.XX | 0.XX|    XX.X    |   XX.X    |    XX     |
| v1 (+CrossAttn)|  X.XX    | 0.XX | 0.XX|    XX.X    |   XX.X    |    XX     |
| v2 (+Percept) |   X.XX    | 0.XX | 0.XX|    XX.X    |   XX.X    |    XX     |
| v3 (+Texture) |   X.XX    | 0.XX | 0.XX|    XX.X    |   XX.X    |    XX     |
| v4 (FINAL)    |   X.XX    | 0.XX | 0.XX|    XX.X    |   XX.X    |    XX     |
```

**Task 5.2: Ablation Study**
Test combinations to understand contribution:
- [ ] Base only
- [ ] Base + Cross-attention
- [ ] Base + Perceptual loss
- [ ] Base + Texture branch
- [ ] Base + Global-local
- [ ] All combinations (2^4 = 16 variants)

**Task 5.3: Qualitative Analysis**
- [ ] Generate visual comparisons (grid of all versions)
- [ ] Failure case analysis
- [ ] Cloud density analysis (light/medium/heavy clouds)
- [ ] Different terrain types (urban/forest/water/etc.)

**Task 5.4: Documentation**

Create comprehensive documentation:

1. **Technical Report**: `results/TECHNICAL_REPORT.md`
   - Architecture details
   - Training procedures
   - Hyperparameters
   - Ablation results
   - Computational analysis

2. **Usage Guide**: `README_ENHANCED.md`
   - How to use each model version
   - Installation instructions
   - Inference examples
   - Training from scratch

3. **Paper Draft**: `paper/enhanced_glf_cr.tex`
   - Introduction (build on GLF-CR)
   - Related work
   - Methodology (each enhancement)
   - Experiments and results
   - Conclusion

**Task 5.5: Model Release**
```python
# Create model zoo
models/
├── glf_cr_baseline.pth
├── glf_cr_v1_cross_attn.pth
├── glf_cr_v2_perceptual.pth
├── glf_cr_v3_texture.pth
└── glf_cr_v4_final.pth

# Create inference script
python inference.py \
    --model v4 \
    --checkpoint models/glf_cr_v4_final.pth \
    --input_cloudy images/cloudy.tif \
    --input_sar images/sar.tif \
    --output images/restored.tif
```

**Deliverables**:
- ✅ Complete evaluation results
- ✅ Ablation study completed
- ✅ Documentation written
- ✅ Model zoo created
- ✅ Paper draft ready

---

## 📊 Expected Overall Improvements

### Summary Table:
```
┌─────────────────┬──────────────┬──────────────┬───────────────┐
│ Metric          │ Baseline     │ Final (v4)   │ Improvement   │
├─────────────────┼──────────────┼──────────────┼───────────────┤
│ PSNR (dB)       │   XX.XX      │  XX.XX       │  +5.0 ~ +7.0  │
│ SSIM            │   0.XXXX     │  0.XXXX      │  +0.10 ~ 0.15 │
│ MAE             │   0.XXX      │  0.XXX       │  -20% ~ -30%  │
│ Params (M)      │   XX.X       │  XX.X        │  +40% ~ 60%   │
│ FLOPs (G)       │   XX.X       │  XX.X        │  +80% ~ 120%  │
│ Inference (ms)  │   XX         │  XX          │  +100% ~ 150% │
└─────────────────┴──────────────┴──────────────┴───────────────┘
```

### Phase-by-Phase Gains:
```
Baseline ─────> v1 (+1.5-2.0 dB) ─────> v2 (+1.0-1.5 dB) ─────> 
        v3 (+0.5-1.0 dB) ─────> v4 (+1.5-2.5 dB) = FINAL

Total Expected: +5.0 ~ +7.0 dB PSNR
                +0.10 ~ 0.15 SSIM
```

---

## 🔧 Development Best Practices

### 1. Version Control
```bash
# Always work on feature branches
git checkout -b feature/cross-modal-attention
git commit -m "Add cross-modal attention module"
git tag v1.0-cross-attn
git push origin v1.0-cross-attn

# Main branch only for stable releases
```

### 2. Experiment Tracking
```python
# Use Weights & Biases or TensorBoard
import wandb

wandb.init(
    project="glf-cr-enhanced",
    name=f"v{version}_{experiment_name}",
    config=config
)

# Log everything
wandb.log({
    "train/psnr": psnr,
    "train/ssim": ssim,
    "train/loss": loss
})
```

### 3. Reproducibility
```python
# Always set seeds
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Save all hyperparameters
config = {
    'seed': 42,
    'lr': 1e-4,
    'batch_size': 8,
    # ... all settings
}
torch.save(config, 'config.pkl')
```

### 4. Testing
```python
# Unit tests for each module
def test_cross_modal_attention():
    model = CrossModalAttention(dim=96)
    optical = torch.randn(2, 64, 96)
    sar = torch.randn(2, 64, 96)
    out_opt, out_sar = model(optical, sar)
    assert out_opt.shape == optical.shape
    assert out_sar.shape == sar.shape

# Run tests before committing
pytest tests/
```

---

## 📅 Detailed Timeline

```
Week 1: [████████████████████] Phase 0: Baseline & Setup
Week 2: [████████████████░░░░] Phase 1: Cross-Modal (Part 1)
Week 3: [████████████████████] Phase 1 & 2: Cross-Modal + Perceptual
Week 4: [████████████████████] Phase 2: Perceptual Loss (Complete)
Week 5: [████████████████░░░░] Phase 3: Texture Branch (Part 1)
Week 6: [████████████████████] Phase 3: Texture Branch (Complete)
Week 7: [████████████████░░░░] Phase 4: Global-Local (Part 1)
Week 8: [████████████████████] Phase 4: Global-Local (Complete)
Week 9: [████████████████░░░░] Phase 5: Evaluation
Week 10:[████████████████████] Phase 5: Documentation & Release
```

---

## 🎯 Success Metrics

### Quantitative:
- ✅ PSNR improvement: ≥ +5.0 dB
- ✅ SSIM improvement: ≥ +0.10
- ✅ Inference time: ≤ 2.5x baseline
- ✅ All phases completed on time

### Qualitative:
- ✅ Visually superior cloud removal
- ✅ Better texture preservation
- ✅ Improved handling of thick clouds
- ✅ More natural color reproduction

### Process:
- ✅ Clean, modular code
- ✅ Comprehensive documentation
- ✅ Reproducible experiments
- ✅ Proper version control

---

## 🚀 Getting Started

```bash
# 1. Clone and setup
git clone <repo>
cd GLF-CR
git checkout -b glf-cr-transformer-enhancements

# 2. Create environment
conda create -n glf-cr-enhanced python=3.8
conda activate glf-cr-enhanced
pip install -r requirements.txt

# 3. Run baseline
python codes/test_CR.py --checkpoint ckpt/CR_net.pth

# 4. Start Phase 1
# Follow tasks in PHASE 1 section above

# 5. Track progress
# Update this file with ✅ as tasks complete
```

---

## 📞 Support & Questions

- **Technical Issues**: Open GitHub issue
- **Architecture Questions**: Check `transformer_enhancement_proposal.md`
- **Training Problems**: See troubleshooting section in technical report

---

## 📝 Notes

- This plan is flexible - adjust timeline based on results
- Each phase can be extended if needed
- Early stopping is okay if target metrics achieved
- Document all deviations from the plan
- Maintain GLF-CR identity - we're enhancing, not replacing!

---

**Remember**: We're building a **hybrid model based on GLF-CR**, not replacing it. Every enhancement should complement the original architecture, not overshadow it. The goal is "GLF-CR++", not "New Model".

🎯 **Core Principle**: Preserve GLF-CR's strengths (dynamic fusion, RDN architecture) while adding transformer capabilities for better performance.

---

**Last Updated**: November 19, 2025  
**Version**: 1.0  
**Status**: Ready to Begin Phase 0
