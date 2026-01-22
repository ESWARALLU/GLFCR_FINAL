import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch

from dataloader import AlignedDataset, get_train_val_test_filelists
from net_CR_RDN import RDN_residual_CR
from metrics import PSNR, SSIM


def load_model(checkpoint_path: str, crop_size: int):
    """Create network and load weights from either training checkpoints or legacy ckpt.
    Supports:
      - Training checkpoints saved by train_CR.py: contains 'model_state_dict'
      - Legacy pretrained: ../ckpt/CR_net.pth with key 'network'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RDN_residual_CR(crop_size).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint with tolerance for formats
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'network' in ckpt:
            state_dict = ckpt['network']

    if state_dict is None:
        # If a raw state_dict was saved
        state_dict = ckpt

    # In case it was saved from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    return net, device


def tensor_to_image(t: torch.Tensor):
    """Convert a BCHW or CHW tensor in [0,1] or [0,255] to PIL Image (RGB)."""
    # Expect shape [B, C, H, W] or [C, H, W]
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu()
    # Clamp to [0,1]
    t = t.float().clamp(0, 1)
    # Channel handling: if >3, take first 3 bands; if 1, repeat to 3
    if t.size(0) > 3:
        t = t[:3, :, :]
    elif t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    # Convert to HWC and uint8
    img = (t.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(img)


def test_single_batch(net, device, opts):
    """Run inference and metrics on the test split (all batches unless single_batch flag)."""
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    dataset = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    pred_dir = Path('../results/predicted')
    pred_dir.mkdir(parents=True, exist_ok=True)

    results_per_image = []
    total_psnr = 0.0
    total_ssim = 0.0
    num_items = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            cloudy = batch['cloudy_data'].to(device)
            cloudfree = batch['cloudfree_data'].to(device)
            sar = batch['SAR_data'].to(device)
            file_names = batch['file_name']

            pred = net(cloudy, sar)

            # Compute per-sample metrics to avoid broadcasting one value to all images
            for i, fn in enumerate(file_names):
                psnr_i = PSNR(pred[i:i+1], cloudfree[i:i+1])
                ssim_i = SSIM(pred[i:i+1], cloudfree[i:i+1]).item()

                results_per_image.append({
                    'image': fn,
                    'psnr': float(psnr_i),
                    'ssim': float(ssim_i),
                })

                total_psnr += psnr_i
                total_ssim += ssim_i
                num_items += 1

                # Save prediction per image
                img = tensor_to_image(pred[i:i+1])
                base = os.path.splitext(os.path.basename(fn))[0]
                out_path = pred_dir / f'{base}_pred.png'
                img.save(out_path)

            if getattr(opts, 'single_batch', False):
                break

    avg_psnr = total_psnr / max(1, num_items)
    avg_ssim = total_ssim / max(1, num_items)

    return avg_psnr, avg_ssim, results_per_image


def save_results(opts, avg_psnr, avg_ssim, results_per_image):
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_entry = {
        'timestamp': timestamp,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': opts.model_name,
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'num_images': len(results_per_image),
        'notes': opts.notes,
        'per_image_results': results_per_image,
        'checkpoint': opts.checkpoint,
        'batch_sz': opts.batch_sz,
        'single_batch': True,
    }

    result_file = os.path.join(results_dir, f'result_{opts.model_name}_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(result_entry, f, indent=4)

    history_file = os.path.join(results_dir, 'results_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    history.append(result_entry)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

    summary_file = os.path.join(results_dir, 'results_summary.csv')
    write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
    with open(summary_file, 'a') as f:
        if write_header:
            f.write('Timestamp,Model,PSNR,SSIM,Num_Images,Notes\n')
        f.write(f'{timestamp},{opts.model_name},{avg_psnr:.4f},{avg_ssim:.4f},{len(results_per_image)},"{opts.notes}"\n')

    return result_file, history_file, summary_file


def main():
    parser = argparse.ArgumentParser()
    # Keep parity with test_CR.py
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for testing (single batch will run)')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)

    parser.add_argument('--model_name', type=str, default='custom_single_batch', help='Model label for logging')
    parser.add_argument('--notes', type=str, default='single-batch evaluation', help='Additional notes')
    parser.add_argument('--single_batch', action='store_true', help='Run only one batch (debug)')

    # New: checkpoint path to evaluate
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth', help='Path to model checkpoint to evaluate')

    opts = parser.parse_args()

    # Prepare model
    net, device = load_model(opts.checkpoint, opts.crop_size)

    print(f"\n{'='*60}")
    print(f"Testing (single batch) Model: {opts.model_name}")
    print(f"Checkpoint: {opts.checkpoint}")
    print(f"{'='*60}\n")

    avg_psnr, avg_ssim, results_per_image = test_single_batch(net, device, opts)

    print(f"\n{'='*60}")
    print(f"Single-Batch Results:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Images in batch: {len(results_per_image)}")
    print(f"{'='*60}\n")

    result_file, history_file, summary_file = save_results(opts, avg_psnr, avg_ssim, results_per_image)

    print("Results saved to:")
    print(f"  - {result_file}")
    print(f"  - {history_file}")
    print(f"  - {summary_file}\n")


if __name__ == '__main__':
    main()
