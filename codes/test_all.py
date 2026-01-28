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
def test_dataset_split(CR_net, opts, filelist, split_name='test', model_name='RDN', lpips_fn=None):
    """Test the model on a specific dataset split (train/val/test)
    Args:
        CR_net: The model to test
        opts: Configuration options
        filelist: List of files to test on
        split_name: Name of the split ('train', 'val', or 'test')
        model_name: 'RDN' or 'CrossAttention'
        lpips_fn: Pre-initialized LPIPS model (optional)
    Returns:
        Dictionary containing average metrics and per-image results
    """
    
    data = AlignedDataset(opts, filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=1,               # FORCE batch=1 (model limitation)
        shuffle=False,
        num_workers=0,              # Avoid worker issues
        pin_memory=True
    )

    dataset_size = len(filelist)
    print(f"\n{'='*60}")
    print(f"Testing {split_name.upper()} set: {dataset_size} images")
    print(f"{'='*60}")

    total_psnr = 0.0
    total_ssim = 0.0
    total_sam = 0.0
    total_rmse = 0.0
    total_lpips = 0.0
    results_per_image = []
    processed_images = 0

    # Use mininterval to avoid spamming logs in Kaggle
    iterator = tqdm(dataloader, total=len(dataloader), desc=f'Testing {split_name}', mininterval=10.0) if tqdm else dataloader

    with torch.no_grad():
        for inputs in iterator:
            cloudy_data = inputs['cloudy_data'].cuda()
            cloudfree_data = inputs['cloudfree_data'].cuda()
            SAR_data = inputs['SAR_data'].cuda()
            file_names = inputs['file_name']

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
                    # Grayscale to RGB
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
                "rmse": rmse_val,
                "lpips": lpips_val
            })

            processed_images += 1

            if tqdm:
                iterator.set_postfix({
                    "PSNR": f"{psnr_val:.3f}",
                    "SSIM": f"{ssim_val:.3f}",
                    "SAM": f"{sam_val:.3f}",
                    "RMSE": f"{rmse_val:.4f}",
                    "LPIPS": f"{lpips_val:.4f}",
                    "Done": processed_images
                })

    avg_psnr = total_psnr / processed_images
    avg_ssim = total_ssim / processed_images
    avg_sam = total_sam / processed_images
    avg_rmse = total_rmse / processed_images
    avg_lpips = total_lpips / processed_images

    print(f"\n{split_name.upper()} Set Results:")
    print(f"  PSNR:  {avg_psnr:.4f} dB")
    print(f"  SSIM:  {avg_ssim:.4f}")
    print(f"  SAM:   {avg_sam:.4f} deg")
    print(f"  RMSE:  {avg_rmse:.5f}")
    print(f"  LPIPS: {avg_lpips:.5f}")
    print(f"  Total Images: {processed_images}")

    return {
        "split_name": split_name,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_sam": avg_sam,
        "avg_rmse": avg_rmse,
        "avg_lpips": avg_lpips,
        "num_images": processed_images,
        "per_image": results_per_image
    }


##########################################################
def test_all_splits(CR_net, opts, model_name='RDN'):
    """Test the model on ALL dataset splits (train, val, test)
    Args:
        CR_net: The model to test
        opts: Configuration options
        model_name: 'RDN' or 'CrossAttention'
    Returns:
        Dictionary containing results for all splits
    """
    # Get all filelists
    train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    
    print("\n" + "="*60)
    print("COMPLETE DATASET EVALUATION")
    print(f"Total Dataset Size: {len(train_filelist) + len(val_filelist) + len(test_filelist)} images")
    print(f"  - Training:   {len(train_filelist)} images")
    print(f"  - Validation: {len(val_filelist)} images")
    print(f"  - Test:       {len(test_filelist)} images")
    print("="*60)
    
    # Initialize LPIPS model once for all splits
    lpips_fn = None
    if lpips:
        try:
            # use alex net as it is standard for LPIPS metric
            lpips_fn = lpips.LPIPS(net='alex').cuda()
            lpips_fn.eval()
            print("LPIPS model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LPIPS: {e}")
    
    all_results = {}
    
    # Test on each split
    if len(train_filelist) > 0:
        all_results['train'] = test_dataset_split(
            CR_net, opts, train_filelist, 'train', model_name, lpips_fn
        )
    
    if len(val_filelist) > 0:
        all_results['val'] = test_dataset_split(
            CR_net, opts, val_filelist, 'val', model_name, lpips_fn
        )
    
    if len(test_filelist) > 0:
        all_results['test'] = test_dataset_split(
            CR_net, opts, test_filelist, 'test', model_name, lpips_fn
        )
    
    # Calculate overall statistics across all splits
    total_images = sum(result['num_images'] for result in all_results.values())
    
    # Weighted average based on number of images in each split
    weighted_psnr = sum(result['avg_psnr'] * result['num_images'] for result in all_results.values()) / total_images
    weighted_ssim = sum(result['avg_ssim'] * result['num_images'] for result in all_results.values()) / total_images
    weighted_sam = sum(result['avg_sam'] * result['num_images'] for result in all_results.values()) / total_images
    weighted_rmse = sum(result['avg_rmse'] * result['num_images'] for result in all_results.values()) / total_images
    weighted_lpips = sum(result['avg_lpips'] * result['num_images'] for result in all_results.values()) / total_images
    
    all_results['overall'] = {
        "avg_psnr": weighted_psnr,
        "avg_ssim": weighted_ssim,
        "avg_sam": weighted_sam,
        "avg_rmse": weighted_rmse,
        "avg_lpips": weighted_lpips,
        "total_images": total_images
    }
    
    print("\n" + "="*60)
    print("OVERALL RESULTS (Weighted Average Across All Splits)")
    print("="*60)
    print(f"  PSNR:  {weighted_psnr:.4f} dB")
    print(f"  SSIM:  {weighted_ssim:.4f}")
    print(f"  SAM:   {weighted_sam:.4f} deg")
    print(f"  RMSE:  {weighted_rmse:.5f}")
    print(f"  LPIPS: {weighted_lpips:.5f}")
    print(f"  Total Images: {total_images}")
    print("="*60)
    
    return all_results


##########################################################
def main():
    parser = argparse.ArgumentParser(description="Test model on complete dataset (all splits)")

    parser.add_argument('--model_type', type=str, default='RDN',
                        choices=['RDN', 'CrossAttention'],
                        help='Model type to test')
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)

    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--notes', type=str, default='Complete dataset evaluation')
    
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/results',
                        help='Directory to save results')

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
    print("COMPLETE DATASET TEST - ALL SPLITS")
    print("Using single GPU (DataParallel disabled)")
    print("="*60)

    # Load checkpoint FIRST to determine model type
    print(f"\nLoading checkpoint: {opts.checkpoint_path}")
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
    if opts.model_type == 'RDN':  # Only check if default or user specified RDN
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

    print("\n" + "="*60)
    print(f"Testing Model: {opts.model_type} - {opts.model_name}")
    print(f"Input Data: {opts.input_data_folder}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print(f"Checkpoint: {opts.checkpoint_path}")
    print("="*60)
    
    # Test on all splits
    all_results = test_all_splits(CR_net, opts, model_name=opts.model_type)

    # Save results
    os.makedirs(opts.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(opts.output_dir, f"complete_test_results_{opts.model_name}_{timestamp}.json")
    
    # Prepare summary for JSON
    summary = {
        "timestamp": timestamp,
        "model": opts.model_name,
        "model_type": opts.model_type,
        "checkpoint": opts.checkpoint_path,
        "data_csv": opts.data_list_filepath,
        "notes": opts.notes,
        "splits": all_results
    }
    
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n\nResults saved to: {out_json}")
    
    # Also save a summary CSV for easy viewing
    csv_path = os.path.join(opts.output_dir, f"complete_test_summary_{opts.model_name}_{timestamp}.csv")
    with open(csv_path, "w", newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Split', 'Images', 'PSNR (dB)', 'SSIM', 'SAM (deg)', 'RMSE', 'LPIPS'])
        
        for split_name in ['train', 'val', 'test']:
            if split_name in all_results:
                r = all_results[split_name]
                writer.writerow([
                    split_name.upper(),
                    r['num_images'],
                    f"{r['avg_psnr']:.4f}",
                    f"{r['avg_ssim']:.4f}",
                    f"{r['avg_sam']:.4f}",
                    f"{r['avg_rmse']:.5f}",
                    f"{r['avg_lpips']:.5f}"
                ])
        
        # Overall weighted average
        r = all_results['overall']
        writer.writerow([
            'OVERALL',
            r['total_images'],
            f"{r['avg_psnr']:.4f}",
            f"{r['avg_ssim']:.4f}",
            f"{r['avg_sam']:.4f}",
            f"{r['avg_rmse']:.5f}",
            f"{r['avg_lpips']:.5f}"
        ])
    
    print(f"Summary CSV saved to: {csv_path}")
    print("\n" + "="*60)
    print("COMPLETE DATASET TESTING FINISHED")
    print("="*60)


if __name__ == "__main__":
    main()
