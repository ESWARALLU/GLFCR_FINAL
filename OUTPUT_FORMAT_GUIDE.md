# Cloud Removal Model Output Format Guide

## Understanding the 13-Band TIFF Output

### The Problem: "Why does my TIFF have 13 separate images?"

When you save a multi-band satellite image as a TIFF file, viewers often display it as multiple "pages" or "separate images". This is **completely normal and correct** behavior for multi-band GeoTIFFs.

### Model Output Structure

The CloudRemovalCrossAttention model outputs:
- **Shape**: (13, H, W) where:
  - 13 = Sentinel-2 bands (B1, B2, ..., B12)
  - H = Height (same as input)
  - W = Width (same as input)
- **Value Range**: [0, 10000] (reflectance values)
- **Data Type**: float32

### Spatial Dimensions: "Should be 10x10 not 1x1"

**The model preserves spatial dimensions:**
- If your input is (13, 256, 256), output is (13, 256, 256)
- If your input is (13, 512, 512), output is (13, 512, 512)
- There is NO downsampling or upsampling in the model

The output spatial resolution matches the INPUT resolution exactly.

### Reading the Multi-Band TIFF in Python

```python
import tifffile
import numpy as np

# Read the 13-band TIFF
data = tifffile.imread('output_13bands.tif')
print(data.shape)  # Output: (13, H, W)

# Access individual bands
band1 = data[0]  # First band
band_nir = data[7]  # NIR band (approximate)

# Create RGB composite (bands 3, 2, 1 = Red, Green, Blue)
rgb = np.stack([data[2], data[1], data[0]], axis=0)
print(rgb.shape)  # Output: (3, H, W)
```

### Output Files Generated

1. **output_13bands.tif** (or output_cloudremoved.tif)
   - Full 13-band output from the model
   - This is the actual cloud-removed prediction
   - Use this for all analysis and metrics

2. **output_rgb.tif** (if available)
   - RGB composite (3 bands only)
   - Easier to view in standard image viewers
   - Derived from: R=Band3, G=Band2, B=Band1

3. **output_rgb.png**
   - PNG visualization of RGB composite
   - Quick preview without specialized tools

4. **metrics.txt**
   - Quality metrics if reference image found:
     - PSNR (dB)
     - SSIM (0-1)
     - SAM (degrees)
     - RMSE

### Model Fusion Verification

To verify the model is correctly fusing SAR and Optical data:

1. **Check output shape** - should match input shape:
   ```python
   data = tifffile.imread('output_13bands.tif')
   assert data.shape[0] == 13, "Should have 13 bands"
   assert data.shape[1] == input_height, "Height should match input"
   assert data.shape[2] == input_width, "Width should match input"
   ```

2. **Check value range** - should be [0, 10000]:
   ```python
   assert data.min() >= 0, "Min value should be >= 0"
   assert data.max() <= 10000, "Max value should be <= 10000"
   ```

3. **Compare with input** - look for cloud removal:
   ```python
   import matplotlib.pyplot as plt
   fig, axes = plt.subplots(1, 3)
   axes[0].imshow(cloudy_optical[2])  # Cloudy (RGB)
   axes[1].imshow(cloudfree_optical[2])  # Cloud-free reference
   axes[2].imshow(output[2])  # Model prediction
   plt.show()
   ```

### Sentinel-2 Band Reference

| Index | Band | Name | Wavelength (nm) |
|-------|------|------|-----------------|
| 0 | B1 | Coastal aerosol | 443 |
| 1 | B2 | Blue | 490 |
| 2 | B3 | Green | 560 |
| 3 | B4 | Red | 665 |
| 4-7 | B5-B8 | Vegetation/NIR | 705-842 |
| 8-12 | B8A, B11, B12, etc | Extended bands | 865+ |

### Troubleshooting

**Q: "Why is output showing as 13 separate images in my viewer?"**
A: This is normal for multi-band TIFFs. Your viewer is correctly showing all 13 bands. Use `output_rgb.tif` or `output_rgb.png` for quick preview.

**Q: "Output dimensions don't match input?"**
A: Check if there's a dimension mismatch error - the updated test_image.py will raise an exception if shapes don't match.

**Q: "Values are outside [0, 10000] range?"**
A: Output is clipped to [0, 10000]. Check the printed output range in test_image.py logs.

**Q: "How do I know the model is working correctly?"**
A: Compare metrics (PSNR, SSIM) with reference image, or visually compare RGB composites.

### Example Usage

```bash
python test_image.py \
  --image_path /path/to/cloudy_image.tif \
  --sar_path /path/to/sar_image.tif \
  --cloudfree_path /path/to/reference_image.tif \
  --model_checkpoint /path/to/checkpoint.pth \
  --output_dir /output/folder
```

### Key Points to Remember

✅ **Correct behaviors:**
- 13 bands stored as separate TIFF pages
- Output shape = Input shape (no downsampling)
- Value range [0, 10000]
- Multiple output formats provided for different uses

❌ **Error conditions:**
- Output shape ≠ Input shape (will raise exception)
- Values outside [0, 10000]
- Missing dimension information
