import os
import torch
import argparse
from datetime import datetime
import json
from pathlib import Path
from PIL import Image

from metrics import PSNR, SSIM

from dataloader import AlignedDataset, get_train_val_test_filelists

from net_CR_RDN import RDN_residual_CR

##########################################################
def tensor_to_image(t: torch.Tensor):
    # Convert a BCHW/CHW tensor in [0,1] to PIL RGB image
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float().clamp(0, 1)
    if t.size(0) > 3:
        t = t[:3, :, :]
    elif t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    img = (t.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(img)


def test(CR_net, opts, model_name='RDN'):
    """
    Test the cloud removal network
    Args:
        CR_net: The model to test
        opts: Configuration options
        model_name: 'RDN' or 'CrossAttention'
    """

    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    
    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz, shuffle=False, num_workers=4)

    iters = 0
    total_psnr = 0
    total_ssim = 0
    results_per_image = []
    
    for inputs in dataloader:

        cloudy_data = inputs['cloudy_optical'].cuda()
        cloudfree_data = inputs['cloudfree_optical'].cuda()
        SAR_data = inputs['sar'].cuda()
        file_name = inputs['file_name'][0]

        # Handle different model forward signatures
        if model_name == 'CrossAttention':
            pred_cloudfree_data = CR_net(cloudy_data, SAR_data)
        else:
            pred_cloudfree_data = CR_net(cloudy_data, SAR_data)
       
        psnr_13 = PSNR(pred_cloudfree_data, cloudfree_data)
        ssim_13 = SSIM(pred_cloudfree_data, cloudfree_data).item()
        
        total_psnr += psnr_13
        total_ssim += ssim_13
        
        results_per_image.append({
            'image': file_name,
            'psnr': float(psnr_13),
            'ssim': float(ssim_13)
        })

        # Save predicted image(s)
        pred_dir = Path('../results/predicted')
        pred_dir.mkdir(parents=True, exist_ok=True)
        img = tensor_to_image(pred_cloudfree_data)
        base = os.path.splitext(os.path.basename(file_name))[0]
        out_path = pred_dir / f'{base}_pred_baseline.png'
        img.save(out_path)
        print(f"Saved predicted baseline image: {out_path}")

        print(iters, '  psnr_13:', format(psnr_13,'.4f'), '  ssim_13:', format(ssim_13,'.4f'))
        iters += 1

        # Optional: if user wants same single image comparison, stop after first batch
        if getattr(opts, 'single_batch', False):
            break
    
    # Calculate averages
    avg_psnr = total_psnr / iters if iters > 0 else 0
    avg_ssim = total_ssim / iters if iters > 0 else 0
    
    return avg_psnr, avg_ssim, results_per_image
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, default='RDN',
                        choices=['RDN', 'CrossAttention'],
                        help='Model type to test')
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data') 
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2) # only useful when is_use_cloudmask=True
    parser.add_argument('--model_name', type=str, default='baseline', help='Model variant name for tracking')
    parser.add_argument('--notes', type=str, default='', help='Additional notes about this run')
    parser.add_argument('--single_batch', action='store_true', help='Run only a single batch to match custom tester')

    opts = parser.parse_args()

    # Load model based on type
    if opts.model_type == 'CrossAttention':
        from net_CR_CrossAttention import CloudRemovalCrossAttention
        CR_net = CloudRemovalCrossAttention().cuda()
    else:
        # Default RDN model
        CR_net = RDN_residual_CR(opts.crop_size).cuda()
    
    # Load checkpoint
    checkpoint = torch.load('../ckpt/CR_net.pth', weights_only=False)
    
    # Handle different checkpoint formats
    if 'network' in checkpoint:
        CR_net.load_state_dict(checkpoint['network'])
    elif 'model_state_dict' in checkpoint:
        CR_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        CR_net.load_state_dict(checkpoint)

    CR_net.eval()
    for _,param in CR_net.named_parameters():
        param.requires_grad = False

    print(f"\n{'='*60}")
    print(f"Testing Model: {opts.model_type} - {opts.model_name}")
    print(f"{'='*60}\n")
    
    avg_psnr, avg_ssim, results_per_image = test(CR_net, opts, model_name=opts.model_type)
    
    print(f"\n{'='*60}")
    print(f"Average Results:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_entry = {
        'timestamp': timestamp,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': opts.model_name,
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'num_images': len(results_per_image),
        'notes': opts.notes,
        'per_image_results': results_per_image
    }
    
    # Save individual run result
    result_file = os.path.join(results_dir, f'result_{opts.model_name}_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(result_entry, f, indent=4)
    
    # Append to history file
    history_file = os.path.join(results_dir, 'results_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(result_entry)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Create/update summary CSV for easy viewing
    summary_file = os.path.join(results_dir, 'results_summary.csv')
    with open(summary_file, 'a') as f:
        if os.path.getsize(summary_file) == 0 if os.path.exists(summary_file) else True:
            f.write('Timestamp,Model,PSNR,SSIM,Num_Images,Notes\n')
        f.write(f'{timestamp},{opts.model_name},{avg_psnr:.4f},{avg_ssim:.4f},{len(results_per_image)},"{opts.notes}"\n')
    
    print(f"Results saved to:")
    print(f"  - {result_file}")
    print(f"  - {history_file}")
    print(f"  - {summary_file}\n")

if __name__ == "__main__":
    main()
    