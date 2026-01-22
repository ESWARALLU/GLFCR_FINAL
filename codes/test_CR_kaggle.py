import os
import torch
import torch.nn as nn
import argparse
import json
from datetime import datetime

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from metrics import PSNR, SSIM, SAM, RMSE
from dataloader import AlignedDataset, get_train_val_test_filelists
from net_CR_RDN import RDN_residual_CR

try:
    import lpips
except ImportError:
    print("LPIPS library not found. Please install it with 'pip install lpips'")
    lpips = None


##########################################################
def test(CR_net, opts, model_name='RDN'):
    """Test the model
    Args:
        CR_net: The model to test
        opts: Configuration options
        model_name: 'RDN' or 'CrossAttention'
    """
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=1,               # FORCE batch=1 (model limitation)
        shuffle=False,
        num_workers=0,              # Avoid worker issues
        pin_memory=True
    )

    test_set_size = len(test_filelist)
    print(f"Test set size: {test_set_size} images")

    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0
    total_sam = 0.0
    total_rmse = 0.0
    total_lpips = 0.0  # Added LPIPS accumulator
    results_per_image = []
    processed_images = 0

    # Use mininterval to avoid spamming logs in Kaggle
    iterator = tqdm(dataloader, total=len(dataloader), desc='Testing', mininterval=10.0) if tqdm else dataloader

    # Initialize LPIPS model
    lpips_fn = None
    if lpips:
        try:
            # use alex net as it is standard for LPIPS metric
            lpips_fn = lpips.LPIPS(net='alex').cuda()
            lpips_fn.eval()
        except Exception as e:
            print(f"Failed to initialize LPIPS: {e}")

    with torch.no_grad():
        for inputs in iterator:
            cloudy_data = inputs['cloudy_data'].cuda()
            cloudfree_data = inputs['cloudfree_data'].cuda()
            SAR_data = inputs['SAR_data'].cuda()
            file_names = inputs['file_name']

            # Initialize LPIPS if needed (lazy init to avoid loading on cpu if not used)
            # Better to pass it in but for now we init here or outside
            # Actually, init outside loop is better for performance.
            # But let's check if we can pass it. 
            # For simplicity, I will assume it's initialized effectively or lightweight enough?
            # No, LPIPS loads a model. It MUST be initialized outside the loop.
            # I'll correct this in the next chunk logic.


            # Handle different model forward signatures
            if model_name == 'CrossAttention':
                pred = CR_net(cloudy_data, SAR_data)
            else:
                pred = CR_net(cloudy_data, SAR_data)

            psnr_val = PSNR(pred, cloudfree_data)
            ssim_val = SSIM(pred, cloudfree_data)
            sam_val = SAM(pred, cloudfree_data)
            rmse_val = RMSE(pred, cloudfree_data)

            # LPIPS
            if lpips_fn:
                # lpips takes (N,3,H,W) in range [-1,1] usually? 
                # Our data is likely 0-1 or normalized. 
                # Check metrics.py or dataloader. Usually images are 0-1.
                # lpips.LPIPS expects input in [-1, 1].
                # If existing data is [0, 1], we need to scale.
                # Most pytorch dataloaders for images might be -1 to 1 or 0 to 1.
                # Given metrics.PSNR usually assumes 0-1 or 0-255?
                # metrics.py: PIXEL_MAX = 1. So data is 0-1.
                # Transform 0..1 to -1..1
                pred_norm = pred * 2.0 - 1.0
                target_norm = cloudfree_data * 2.0 - 1.0
                
                # Check channels. LPIPS expects 3 channels.
                if pred.shape[1] == 3:
                    lpips_val = lpips_fn(pred_norm, target_norm)
                elif pred.shape[1] > 3:
                     # Use first 3 channels (RGB)
                    lpips_val = lpips_fn(pred_norm[:, :3, :, :], target_norm[:, :3, :, :])
                else:
                    # Grayscale to RGB?
                    lpips_val = lpips_fn(pred_norm.repeat(1,3,1,1), target_norm.repeat(1,3,1,1))
                
                lpips_val = lpips_val.item()
            else:
                lpips_val = 0.0


            psnr_val = float(psnr_val.item()) if hasattr(psnr_val, "item") else float(psnr_val)
            ssim_val = float(ssim_val.item()) if hasattr(ssim_val, "item") else float(ssim_val)
            sam_val = float(sam_val.item()) if hasattr(sam_val, "item") else float(sam_val)
            rmse_val = float(rmse_val.item()) if hasattr(rmse_val, "item") else float(rmse_val)

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_sam += sam_val
            total_rmse += rmse_val
            total_lpips += lpips_val

            results_per_image.append({
                "image": file_names,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "sam": sam_val,
                "sam": sam_val,
                "rmse": rmse_val,
                "lpips": lpips_val
            })

            processed_images += 1

            if tqdm:
                iterator.set_postfix({
                    "PSNR": f"{psnr_val:.3f}",
                    "SSIM": f"{ssim_val:.3f}",
                    "SAM": f"{sam_val:.3f}",
                    "SAM": f"{sam_val:.3f}",
                    "RMSE": f"{rmse_val:.4f}",
                    "LPIPS": f"{lpips_val:.4f}",
                    "Done": processed_images
                })

    avg_psnr = total_psnr / processed_images
    avg_ssim = total_ssim / processed_images
    avg_sam = total_sam / processed_images
    avg_sam = total_sam / processed_images
    avg_rmse = total_rmse / processed_images
    avg_lpips = total_lpips / processed_images

    return avg_psnr, avg_ssim, avg_sam, avg_rmse, avg_lpips, results_per_image


##########################################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='RDN',
                        choices=['RDN', 'CrossAttention'],
                        help='Model type to test')
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    parser.add_argument('--checkpoint_path', type=str, default='/kaggle/input/checkpoint3/pytorch/default/1/checkpoint_epoch_2.pth')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)

    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--notes', type=str, default='')

    opts = parser.parse_args()

    # Choose CSV properly
    working_csv = '/kaggle/working/data.csv'
    input_csv = os.path.join(opts.input_data_folder, 'data.csv')

    if opts.data_list_filepath == working_csv:
        if os.path.exists(working_csv):
            opts.data_list_filepath = working_csv
        elif os.path.exists(input_csv):
            print(f"Using dataset CSV: {input_csv}")
            opts.data_list_filepath = input_csv
        else:
            raise FileNotFoundError("No CSV available")
    else:
        if not os.path.exists(opts.data_list_filepath):
            raise FileNotFoundError(f"CSV not found: {opts.data_list_filepath}")

    # Model
    print("="*60)
    print("Using single GPU (DataParallel disabled)")
    print("="*60)

    # Load model based on type
    # Load checkpoint FIRST to determine model type
    print(f"Loading checkpoint: {opts.checkpoint_path}")
    try:
        checkpoint = torch.load(
            opts.checkpoint_path,
            map_location="cuda",
            weights_only=False     # REQUIRED FIX for PyTorch 2.6
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "network" in checkpoint:
        state_dict = checkpoint["network"]
    else:
        state_dict = checkpoint

    # Auto-detect model type from keys
    if opts.model_type == 'RDN':  # Only check if default or user specified RDN (to allow override if needed)
        # Check for CrossAttention specific keys
        has_optical = any('optical_encoder' in k for k in state_dict.keys())
        has_cross = any('cross_attn' in k for k in state_dict.keys())
        
        if has_optical or has_cross:
            print("INFO: Auto-detected CrossAttention keys in checkpoint. Switching model_type to 'CrossAttention'.")
            opts.model_type = 'CrossAttention'
        else:
            print("INFO: Detected RDN keys (or no CrossAttention keys). Using 'RDN' model.")

    # Load model based on type (or auto-detected type)
    if opts.model_type == 'CrossAttention':
        from net_CR_CrossAttention import CloudRemovalCrossAttention
        CR_net = CloudRemovalCrossAttention().cuda()
    else:
        # Default RDN model
        CR_net = RDN_residual_CR(opts.crop_size).cuda()
    
    CR_net.eval()
    for p in CR_net.parameters():
        p.requires_grad = False

    CR_net.load_state_dict(state_dict, strict=False)

    print("="*60)
    print(f"Testing Model: {opts.model_type} - {opts.model_name}")
    print(f"Input Data: {opts.input_data_folder}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print(f"Checkpoint: {opts.checkpoint_path}")
    print("="*60)
    
    avg_psnr, avg_ssim, avg_sam, avg_rmse, avg_lpips, results_per_image = test(CR_net, opts, model_name=opts.model_type)

    print("-"*100)
    print("="*60)
    print(f"Average Results:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  SAM:  {avg_sam:.4f} deg")
    print(f"  RMSE: {avg_rmse:.5f}")
    print(f"  LPIPS: {avg_lpips:.5f}")
    print(f"  Total Images: {len(results_per_image)}")
    print("="*60)

    # Save results
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(results_dir, f"results_{opts.model_name}_{timestamp}.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": opts.model_name,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_sam": avg_sam,
            "avg_rmse": avg_rmse,
            "avg_lpips": avg_lpips,
            "num_images": len(results_per_image),
            "per_image": results_per_image
        }, f, indent=4)

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()