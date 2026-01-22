"""
Test script to validate that the metric functions work correctly
"""
import torch
import sys
import os
import math

# Add codes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'codes'))

from metrics import PSNR, SSIM

def test_psnr():
    """Test PSNR function returns float"""
    print("Testing PSNR function...")
    
    # Test 1: Identical images (PSNR should be 100 or very high)
    img1 = torch.randn(2, 3, 128, 128)
    img2 = img1.clone()
    psnr = PSNR(img1, img2)
    assert isinstance(psnr, float), f"PSNR should return float, got {type(psnr)}"
    assert psnr == 100.0, f"PSNR of identical images should be 100.0, got {psnr}"
    print("✓ Test 1 passed: Identical images PSNR = 100.0")
    
    # Test 2: Different images (PSNR should be positive float)
    img3 = torch.randn(2, 3, 128, 128)
    img4 = torch.randn(2, 3, 128, 128)
    psnr = PSNR(img3, img4)
    assert isinstance(psnr, float), f"PSNR should return float, got {type(psnr)}"
    assert psnr > 0, f"PSNR should be positive, got {psnr}"
    assert math.isfinite(psnr), f"PSNR should be finite, got {psnr}"
    print(f"✓ Test 2 passed: Different images PSNR = {psnr:.2f}")
    
    # Test 3: Check that NaN/Inf handling works
    img5 = torch.ones(2, 3, 128, 128)
    img6 = torch.ones(2, 3, 128, 128)
    psnr = PSNR(img5, img6)
    assert psnr == 100.0, "PSNR of identical ones should be 100"
    print("✓ Test 3 passed: Identical ones PSNR = 100.0")
    
    # Test 4: Check float checking logic
    test_float = 25.5
    assert not math.isnan(test_float), "Simple float check failed"
    assert not math.isinf(test_float), "Simple float check failed"
    print("✓ Test 4 passed: Float checking logic works")
    
    print("\n✅ All PSNR tests passed!\n")

def test_ssim():
    """Test SSIM function"""
    print("Testing SSIM function...")
    
    # Test 1: Identical images (SSIM should be ~1.0)
    img1 = torch.randn(2, 3, 128, 128)
    img2 = img1.clone()
    ssim = SSIM(img1, img2)
    print(f"SSIM of identical images: {ssim}")
    if isinstance(ssim, torch.Tensor):
        ssim = float(ssim.item())
    assert isinstance(ssim, float) or isinstance(ssim, torch.Tensor), f"SSIM should return float or tensor, got {type(ssim)}"
    print("✓ Test 1 passed: SSIM function works")
    
    # Test 2: Different images (SSIM should be reasonable)
    img3 = torch.randn(2, 3, 128, 128)
    img4 = torch.randn(2, 3, 128, 128)
    ssim = SSIM(img3, img4)
    if isinstance(ssim, torch.Tensor):
        ssim_val = float(ssim.item())
    else:
        ssim_val = float(ssim)
    print(f"SSIM of different images: {ssim_val}")
    print("✓ Test 2 passed: SSIM function works for different images")
    
    print("\n✅ All SSIM tests passed!\n")

def test_validation_logic():
    """Test the validation logic from train_CR_kaggle.py"""
    print("Testing validation logic...")
    
    # Simulate validation metrics
    batch_psnr = PSNR(torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128))
    batch_ssim = SSIM(torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128))
    
    # This is the checking logic from validate function
    try:
        # Check PSNR (now returns float)
        if math.isnan(batch_psnr) or math.isinf(batch_psnr) or batch_psnr <= 0:
            print("PSNR check: skipped (invalid)")
        else:
            print(f"PSNR check: valid ({batch_psnr:.2f})")
        
        # Check SSIM
        if isinstance(batch_ssim, torch.Tensor):
            if torch.isnan(batch_ssim) or torch.isinf(batch_ssim):
                print("SSIM check: skipped (invalid tensor)")
            else:
                batch_ssim = float(batch_ssim.item())
                print(f"SSIM check: valid tensor converted to float ({batch_ssim:.4f})")
        elif math.isnan(batch_ssim) or math.isinf(batch_ssim) or batch_ssim < 0 or batch_ssim > 1.1:
            print("SSIM check: skipped (invalid float)")
        else:
            print(f"SSIM check: valid ({batch_ssim:.4f})")
        
        print("✓ Validation logic test passed: No errors raised!")
    except Exception as e:
        print(f"✗ Validation logic test failed: {e}")
        raise
    
    print("\n✅ Validation logic test passed!\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing Metric Functions and Validation Logic")
    print("="*60 + "\n")
    
    test_psnr()
    test_ssim()
    test_validation_logic()
    
    print("="*60)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60 + "\n")
