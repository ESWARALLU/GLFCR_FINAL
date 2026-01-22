"""
Diagnostic script to verify model output format and spatial dimensions
Run this after test_image.py to verify that output is correct
"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_output(output_dir='/kaggle/working/images_pred'):
    """
    Diagnose the model output files and verify correctness
    """
    print("\n" + "="*70)
    print("🔍 CLOUD REMOVAL MODEL OUTPUT DIAGNOSTICS")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    print(f"\n📁 Output Directory: {output_dir}")
    
    # Check files
    tiff_files = list(output_dir.glob('*.tif')) + list(output_dir.glob('*.tiff'))
    png_files = list(output_dir.glob('*.png'))
    metrics_file = output_dir / 'metrics.txt'
    
    print(f"\n📋 Files Found:")
    print(f"   TIFF files: {len(tiff_files)}")
    for f in tiff_files:
        print(f"     • {f.name}")
    print(f"   PNG files: {len(png_files)}")
    for f in png_files:
        print(f"     • {f.name}")
    if metrics_file.exists():
        print(f"   Metrics: {metrics_file.name}")
    
    # Analyze 13-band output
    band13_files = [f for f in tiff_files if '13bands' in f.name or 'cloudremoved' in f.name]
    
    if band13_files:
        main_output = band13_files[0]
        print(f"\n🔬 Analyzing Main Output: {main_output.name}")
        
        try:
            data = tifffile.imread(str(main_output))
            print(f"   Shape: {data.shape}")
            
            if len(data.shape) == 3 and data.shape[0] == 13:
                print(f"   ✅ Correct structure: 13 bands")
                print(f"   ✅ Spatial dimensions: {data.shape[1]} × {data.shape[2]}")
                print(f"   ✅ Bands stored as separate TIFF pages (THIS IS NORMAL!)")
                
                # Check value ranges
                print(f"\n   Value Statistics:")
                print(f"   • Min: {data.min():.2f}")
                print(f"   • Max: {data.max():.2f}")
                print(f"   • Mean: {data.mean():.2f}")
                print(f"   • Std: {data.std():.2f}")
                
                # Verify range
                if 0 <= data.min() and data.max() <= 10000:
                    print(f"   ✅ Value range [0, 10000] - CORRECT!")
                else:
                    print(f"   ⚠️  Unexpected value range: [{data.min()}, {data.max()}]")
                
                # Per-band statistics
                print(f"\n   Per-Band Statistics:")
                for i in range(min(5, data.shape[0])):  # Show first 5 bands
                    band = data[i]
                    print(f"   Band {i+1}: min={band.min():.1f}, max={band.max():.1f}, mean={band.mean():.1f}")
                
                # Visualize
                print(f"\n   🎨 Creating visualizations...")
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # RGB composite (bands 3, 2, 1)
                rgb = np.stack([data[2], data[1], data[0]], axis=0)
                rgb_norm = np.clip(rgb / 10000.0, 0, 1)
                axes[0, 0].imshow(np.transpose(rgb_norm, (1, 2, 0)))
                axes[0, 0].set_title('RGB Composite (Bands 3,2,1)')
                axes[0, 0].axis('off')
                
                # NIR band (Band 8, index 7)
                axes[0, 1].imshow(data[7], cmap='viridis')
                axes[0, 1].set_title('NIR Band (Band 8)')
                axes[0, 1].axis('off')
                
                # Band 1 (coastal)
                axes[1, 0].imshow(data[0], cmap='gray')
                axes[1, 0].set_title('Band 1 (Coastal Aerosol)')
                axes[1, 0].axis('off')
                
                # SWIR band (Band 11, index 10)
                axes[1, 1].imshow(data[10], cmap='viridis')
                axes[1, 1].set_title('SWIR Band (Band 11)')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                diag_file = output_dir / 'diagnostics_visualization.png'
                plt.savefig(str(diag_file), dpi=150)
                plt.close()
                print(f"   ✅ Visualization saved: {diag_file.name}")
                
            else:
                print(f"   ⚠️  Unexpected shape: {data.shape}")
                if len(data.shape) == 2:
                    print(f"   ❌ ERROR: Only 2D array - where are the 13 bands?")
                elif data.shape[0] != 13:
                    print(f"   ❌ ERROR: First dimension is {data.shape[0]}, expected 13")
                    
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    # Check RGB output
    rgb_files = [f for f in tiff_files if 'rgb' in f.name.lower()]
    if rgb_files:
        print(f"\n🔬 Analyzing RGB Output: {rgb_files[0].name}")
        try:
            rgb_data = tifffile.imread(str(rgb_files[0]))
            print(f"   Shape: {rgb_data.shape}")
            if len(rgb_data.shape) == 3 and rgb_data.shape[0] == 3:
                print(f"   ✅ Correct RGB structure: 3 bands (R, G, B)")
            else:
                print(f"   ⚠️  Unexpected RGB shape")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check metrics
    if metrics_file.exists():
        print(f"\n📊 Quality Metrics:")
        try:
            with open(metrics_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.strip():
                        print(f"   {line}")
        except Exception as e:
            print(f"   ❌ Error reading metrics: {e}")
    
    # Summary and guidance
    print(f"\n" + "="*70)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\n📚 Key Information:")
    print("  1. 13 bands shown as 'separate images' in TIFF = NORMAL behavior")
    print("  2. Spatial dimensions should match input (preserved through model)")
    print("  3. Value range [0, 10000] matches Sentinel-2 reflectance scale")
    print("  4. Each band represents a different wavelength region")
    print("\n🔧 To read multi-band output in Python:")
    print("  import tifffile")
    print("  data = tifffile.imread('output_13bands.tif')")
    print("  print(data.shape)  # Should show (13, height, width)")
    print("  band_1 = data[0]   # Access individual band")
    print("\n" + "="*70)

if __name__ == '__main__':
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working/images_pred'
    diagnose_output(output_dir)
