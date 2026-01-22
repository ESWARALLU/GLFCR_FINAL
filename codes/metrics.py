import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(img1, img2, mask=None):
    """
    Calculate PSNR between two images.
    Returns a float value for compatibility with validation code.
    """
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)

    # Convert to float if tensor
    if isinstance(mse, torch.Tensor):
        mse_val = mse.item()
    else:
        mse_val = float(mse)
    
    # Handle edge cases
    if mse_val == 0:
        return 100.0
    
    # Handle invalid MSE values (NaN, inf, negative, or extremely large)
    if math.isnan(mse_val) or math.isinf(mse_val) or mse_val < 0 or mse_val > 1e6:
        return 0.0  # Return 0 for invalid cases
    
    PIXEL_MAX = 1
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))
    
    # Final safety check
    if math.isnan(psnr_value) or math.isinf(psnr_value):
        return 0.0
    
    return psnr_value

def RMSE(img1, img2):
    """
    Calculate Root Mean Square Error (RMSE) using PyTorch.
    img1, img2: (B, C, H, W) tensors
    """
    mse = torch.mean((img1 - img2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def SAM(img1, img2):
    """
    Calculate Spectral Angle Mapper (SAM) using PyTorch.
    img1, img2: (B, C, H, W) tensors, values should be normalized (typically 0-1)
    """
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C, N)
    b, c, h, w = img1.shape
    img1_flat = img1.view(b, c, -1)
    img2_flat = img2.view(b, c, -1)
    
    # Calculate dot product along channel dimension
    # sum(A * B, dim=1) -> (B, N)
    dot_product = torch.sum(img1_flat * img2_flat, dim=1)
    
    # Calculate norms
    norm1 = torch.norm(img1_flat, dim=1)
    norm2 = torch.norm(img2_flat, dim=1)
    
    # Avoid division by zero
    valid_mask = (norm1 > 1e-8) & (norm2 > 1e-8)
    
    # Cosine of the angle
    # Clip cos_theta to [-1, 1] to avoid NaNs in acos due to precision errors
    cos_theta = dot_product / (norm1 * norm2 + 1e-10)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Angle in radians
    angles_rad = torch.acos(cos_theta)
    
    # Convert to degrees
    angles_deg = torch.rad2deg(angles_rad)
    
    # Return mean SAM for the batch (masked)
    if valid_mask.sum() > 0:
        sam_val = angles_deg[valid_mask].mean()
    else:
        sam_val = torch.tensor(0.0).to(img1.device)
        
    return sam_val
