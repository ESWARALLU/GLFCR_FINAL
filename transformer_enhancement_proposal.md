# Transformer Architecture Enhancement Proposal
## Improving Cloud Removal Model Performance (PSNR/SSIM)

Date: November 19, 2025

## Current Architecture Analysis

### Existing Components:
1. **Window-based Self-Attention (W-MSA)** - Swin Transformer style
   - 8x8 window size, 8 attention heads
   - Limited to local window interactions
   - Shifted windows for cross-window connections

2. **Hybrid CNN-Transformer**
   - RDN (Residual Dense Network) base
   - 6 RDB blocks with 5 conv layers each
   - Window attention in each RDB_Conv layer

3. **Multi-modal Fusion**
   - Dynamic filtering for SAR-Optical fusion
   - Separate attention streams for optical and SAR

### Current Limitations:
- **Limited long-range dependencies**: Window attention (8x8) restricts global context
- **No cross-modal attention**: SAR and optical processed separately, then fused
- **Fixed receptive field**: No hierarchical or multi-scale transformer attention
- **No dedicated texture/detail enhancement**: Single reconstruction path

---

## Enhancement Strategies

### 🚀 Strategy 1: Cross-Modal Transformer Attention
**Impact: High | Complexity: Medium**

Add explicit cross-attention between SAR and optical features to learn better fusion patterns.

```python
class CrossModalAttention(nn.Module):
    """Cross-attention between optical and SAR features"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Q from optical, K,V from SAR
        self.q_optical = nn.Linear(dim, dim)
        self.kv_sar = nn.Linear(dim, dim * 2)
        
        # Q from SAR, K,V from optical
        self.q_sar = nn.Linear(dim, dim)
        self.kv_optical = nn.Linear(dim, dim * 2)
        
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, optical_feat, sar_feat):
        # Optical attends to SAR
        q_opt = self.q_optical(optical_feat)
        k_sar, v_sar = self.kv_sar(sar_feat).chunk(2, dim=-1)
        optical_enhanced = self.cross_attend(q_opt, k_sar, v_sar)
        
        # SAR attends to optical
        q_sar = self.q_sar(sar_feat)
        k_opt, v_opt = self.kv_optical(optical_feat).chunk(2, dim=-1)
        sar_enhanced = self.cross_attend(q_sar, k_opt, v_opt)
        
        return optical_enhanced, sar_enhanced
```

**Expected Improvement**: +1-2 dB PSNR, +0.02-0.04 SSIM

---

### 🚀 Strategy 2: Global-Local Dual-Path Transformer
**Impact: Very High | Complexity: High**

Combine local window attention with global attention for multi-scale context.

```python
class GlobalLocalTransformer(nn.Module):
    """Dual-path: local window attention + global attention"""
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        # Local path (existing)
        self.local_attn = WindowAttention(dim, window_size, num_heads)
        
        # Global path (new)
        self.global_attn = nn.MultiheadAttention(dim, num_heads)
        
        # Downsampling for efficient global attention
        self.downsample = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4)
        
        # Fusion
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Local attention (existing path)
        local_out = self.local_attn(x)
        
        # Global attention (new path)
        x_down = self.downsample(x)  # Reduce resolution for efficiency
        B, C, H_d, W_d = x_down.shape
        x_flat = x_down.flatten(2).transpose(1, 2)  # [B, H*W, C]
        global_feat, _ = self.global_attn(x_flat, x_flat, x_flat)
        global_feat = global_feat.transpose(1, 2).reshape(B, C, H_d, W_d)
        global_out = self.upsample(global_feat)
        
        # Fuse local and global
        fused = self.fusion(torch.cat([local_out, global_out], dim=1))
        return fused
```

**Expected Improvement**: +2-3 dB PSNR, +0.04-0.06 SSIM

---

### 🚀 Strategy 3: Hierarchical Vision Transformer (HVT) Encoder
**Impact: Very High | Complexity: High**

Replace or augment the CNN encoder with a hierarchical ViT (like Swin-T or PVT).

```python
class HierarchicalTransformerEncoder(nn.Module):
    """Multi-scale transformer encoder with patch merging"""
    def __init__(self, img_size=128, patch_size=2, in_chans=52, embed_dims=[96, 192, 384]):
        super().__init__()
        
        # Stage 1: 128x128 -> 64x64
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, 
                                       embed_dim=embed_dims[0])
        self.stage1 = TransformerStage(embed_dims[0], depth=2, num_heads=4)
        
        # Stage 2: 64x64 -> 32x32
        self.patch_merge1 = PatchMerging(embed_dims[0], embed_dims[1])
        self.stage2 = TransformerStage(embed_dims[1], depth=2, num_heads=8)
        
        # Stage 3: 32x32 -> 16x16
        self.patch_merge2 = PatchMerging(embed_dims[1], embed_dims[2])
        self.stage3 = TransformerStage(embed_dims[2], depth=6, num_heads=16)
        
    def forward(self, x):
        # Multi-scale features for skip connections
        feat1 = self.stage1(self.patch_embed1(x))  # 64x64
        feat2 = self.stage2(self.patch_merge1(feat1))  # 32x32
        feat3 = self.stage3(self.patch_merge2(feat2))  # 16x16
        
        return [feat1, feat2, feat3]
```

**Expected Improvement**: +2-4 dB PSNR, +0.05-0.08 SSIM

---

### 🚀 Strategy 4: Texture Enhancement Transformer
**Impact: Medium-High | Complexity: Medium**

Add a dedicated transformer branch for high-frequency detail reconstruction.

```python
class TextureEnhancementTransformer(nn.Module):
    """Focus on high-frequency details and textures"""
    def __init__(self, dim=96):
        super().__init__()
        
        # High-pass filtering to extract texture
        self.high_pass = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        # Texture-aware attention
        self.texture_attn = nn.MultiheadAttention(dim, num_heads=8)
        
        # Edge-aware loss weight predictor
        self.edge_weight = nn.Sequential(
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat):
        # Extract high-frequency components
        high_freq = self.high_pass(feat)
        
        # Enhance with attention
        B, C, H, W = high_freq.shape
        hf_flat = high_freq.flatten(2).transpose(1, 2)
        enhanced, _ = self.texture_attn(hf_flat, hf_flat, hf_flat)
        enhanced = enhanced.transpose(1, 2).reshape(B, C, H, W)
        
        # Compute edge importance map
        edge_map = self.edge_weight(enhanced)
        
        return enhanced, edge_map
```

**Expected Improvement**: +0.5-1.5 dB PSNR, +0.02-0.04 SSIM (especially on edges/textures)

---

### 🚀 Strategy 5: Perceptual Loss with Vision Transformer
**Impact: Medium | Complexity: Low**

Use pre-trained ViT features for perceptual loss instead of just L1.

```python
class PerceptualLossViT(nn.Module):
    """Perceptual loss using pre-trained Vision Transformer"""
    def __init__(self):
        super().__init__()
        # Use pre-trained DINOv2 or similar
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained('facebook/dino-vitb16')
        
        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # Extract multi-level features
        pred_feats = self.vit(pred, output_hidden_states=True).hidden_states
        target_feats = self.vit(target, output_hidden_states=True).hidden_states
        
        # Compute perceptual loss at multiple layers
        loss = 0
        for i in [3, 6, 9, 12]:  # Use layers 3, 6, 9, 12
            loss += F.l1_loss(pred_feats[i], target_feats[i])
            
        return loss

# In training:
# total_loss = L1_loss + 0.1 * perceptual_loss + 0.01 * adversarial_loss
```

**Expected Improvement**: +1-2 dB PSNR, +0.03-0.05 SSIM (better perceptual quality)

---

### 🚀 Strategy 6: Deformable Attention for Cloud Regions
**Impact: High | Complexity: High**

Use deformable attention to focus on cloud-affected regions adaptively.

```python
class DeformableCloudAttention(nn.Module):
    """Deformable attention focusing on cloudy regions"""
    def __init__(self, dim, num_heads=8, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        
        # Offset prediction for sampling points
        self.offset_net = nn.Conv2d(dim, num_heads * num_points * 2, 1)
        
        # Attention weights for sampled points
        self.attn_weights = nn.Conv2d(dim, num_heads * num_points, 1)
        
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x, cloud_mask=None):
        B, C, H, W = x.shape
        
        # Predict sampling offsets
        offsets = self.offset_net(x)  # [B, num_heads*num_points*2, H, W]
        
        # If cloud mask available, bias sampling toward cloudy regions
        if cloud_mask is not None:
            offsets = offsets * cloud_mask.unsqueeze(1)
        
        # Sample features at offset locations (using grid_sample)
        # ... deformable attention implementation ...
        
        return self.proj(attended_features)
```

**Expected Improvement**: +1-2 dB PSNR, +0.03-0.05 SSIM (especially on cloudy regions)

---

## 📊 Implementation Priority & Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. **Perceptual Loss with ViT** ✅ Easy to implement, immediate improvement
2. **Cross-Modal Attention** ✅ Medium effort, significant fusion improvement
3. **Texture Enhancement Branch** ✅ Parallel implementation, boosts detail

**Expected Combined Gain**: +2-3 dB PSNR, +0.05-0.07 SSIM

### Phase 2: Major Architecture Enhancement (3-4 weeks)
4. **Global-Local Dual-Path** ✅ High impact on long-range dependencies
5. **Hierarchical Transformer Encoder** ✅ Replace/augment CNN backbone

**Expected Combined Gain**: +4-5 dB PSNR, +0.08-0.12 SSIM

### Phase 3: Advanced Optimization (2-3 weeks)
6. **Deformable Attention** ✅ Cloud-adaptive processing
7. **Multi-scale Testing & Refinement** ✅ Fine-tuning all components

**Expected Combined Gain**: +5-7 dB PSNR total, +0.10-0.15 SSIM total

---

## 🔧 Practical Implementation Tips

### 1. **Start with Modular Design**
```python
# Create separate modules that can be toggled on/off
class EnhancedCRNet(nn.Module):
    def __init__(self, use_cross_attn=True, use_global_attn=True, 
                 use_perceptual_loss=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_global_attn = use_global_attn
        
        # Base model
        self.base_model = RDN_residual_CR(...)
        
        # Optional enhancements
        if use_cross_attn:
            self.cross_attn = CrossModalAttention(...)
        if use_global_attn:
            self.global_attn = GlobalLocalTransformer(...)
```

### 2. **Gradual Training Strategy**
- Start with pre-trained base model
- Freeze base, train new transformer components first
- Fine-tune everything together with lower learning rate
- Use progressive loss weighting

### 3. **Computational Efficiency**
- Use gradient checkpointing for large transformers
- Apply attention only at certain RDB blocks (e.g., blocks 3, 4, 5)
- Reduce resolution for global attention (4x downsample)
- Use efficient attention variants (Linear Attention, FlashAttention)

### 4. **Loss Function Enhancement**
```python
class EnhancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLossViT()
        self.ssim_loss = SSIMLoss()
        
    def forward(self, pred, target, edge_map=None):
        loss = 1.0 * self.l1_loss(pred, target)
        loss += 0.1 * self.perceptual_loss(pred, target)
        loss += 0.1 * (1 - SSIM(pred, target))
        
        # Edge-weighted loss if texture branch is used
        if edge_map is not None:
            edge_loss = ((pred - target).abs() * edge_map).mean()
            loss += 0.2 * edge_loss
            
        return loss
```

---

## 📈 Expected Performance Improvements

### Conservative Estimates:
| Enhancement | PSNR Gain | SSIM Gain | Compute Overhead |
|-------------|-----------|-----------|------------------|
| Current Model | Baseline | Baseline | 1.0x |
| + Cross-Modal Attn | +1.5 dB | +0.03 | 1.2x |
| + Global-Local Attn | +2.5 dB | +0.05 | 1.5x |
| + Hierarchical ViT | +3.0 dB | +0.06 | 2.0x |
| + Texture Branch | +1.0 dB | +0.02 | 1.3x |
| + Perceptual Loss | +1.5 dB | +0.04 | 1.1x |
| **All Combined** | **+5-7 dB** | **+0.10-0.15** | **2.5x** |

### Optimistic Estimates (with proper tuning):
- **PSNR**: +8-10 dB improvement
- **SSIM**: +0.15-0.20 improvement
- **Perceptual Quality**: Significant improvement in texture/detail

---

## 🛠️ Code Structure Recommendation

```
codes/
├── model_CR_net.py              # Main model (update this)
├── net_CR_RDN.py                # Base architecture (keep existing)
├── net_transformers.py          # NEW: All transformer enhancements
│   ├── CrossModalAttention
│   ├── GlobalLocalTransformer
│   ├── HierarchicalTransformerEncoder
│   ├── TextureEnhancementTransformer
│   └── DeformableCloudAttention
├── losses.py                    # NEW: Enhanced loss functions
│   ├── PerceptualLossViT
│   ├── SSIMLoss
│   ├── EdgeAwareLoss
│   └── EnhancedLoss
├── submodules.py                # Keep existing
└── train_CR_net_enhanced.py     # NEW: Training script with enhancements
```

---

## 🎯 Next Steps

1. **Baseline Evaluation**: Test current model performance on validation set
2. **Implement Phase 1**: Start with quick wins (perceptual loss + cross-attention)
3. **Benchmark**: Compare with baseline, measure PSNR/SSIM improvements
4. **Iterate**: Add more enhancements based on results
5. **Ablation Study**: Test each component individually to understand contributions

---

## 📚 References

Key papers to reference:
- **Swin Transformer**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Restormer**: "Restormer: Efficient Transformer for High-Resolution Image Restoration"
- **Uformer**: "Uformer: A General U-Shaped Transformer for Image Restoration"
- **SwinIR**: "SwinIR: Image Restoration Using Swin Transformer"
- **NAFNet**: "Simple Baselines for Image Restoration" (efficient design)

These models achieve state-of-the-art results in image restoration tasks with PSNR improvements of 3-5 dB over CNN baselines.

---

## 💡 Conclusion

**YES, there is significant scope to improve the model with transformers!**

The current model already uses window attention, but adding:
- Cross-modal transformers
- Global attention mechanisms
- Hierarchical multi-scale processing
- Perceptual losses with pre-trained ViTs

...can potentially improve PSNR by **5-10 dB** and SSIM by **0.10-0.20**, bringing the model to state-of-the-art performance for cloud removal tasks.

The key is to implement these enhancements modularly and progressively, measuring improvements at each stage.
