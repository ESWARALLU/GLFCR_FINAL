"""
Data quality check script - verify normalization and ranges
"""
import torch
import argparse
from dataloader import AlignedDataset, get_train_val_test_filelists

def check_data_stats():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data') 
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    opts = parser.parse_args()
    
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    print(f"\nFound {len(test_filelist)} test images\n")
    
    data = AlignedDataset(opts, test_filelist)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False)
    
    for idx, inputs in enumerate(dataloader):
        cloudy_data = inputs['cloudy_data']
        cloudfree_data = inputs['cloudfree_data']
        SAR_data = inputs['SAR_data']
        file_name = inputs['file_name'][0]
        
        print(f"Image {idx+1}: {file_name}")
        print(f"  Cloudy optical shape: {cloudy_data.shape}")
        print(f"  Cloudy optical range: [{cloudy_data.min().item():.4f}, {cloudy_data.max().item():.4f}]")
        print(f"  Cloudy optical mean: {cloudy_data.mean().item():.4f}")
        
        print(f"  Cloudfree optical shape: {cloudfree_data.shape}")
        print(f"  Cloudfree optical range: [{cloudfree_data.min().item():.4f}, {cloudfree_data.max().item():.4f}]")
        print(f"  Cloudfree optical mean: {cloudfree_data.mean().item():.4f}")
        
        print(f"  SAR shape: {SAR_data.shape}")
        print(f"  SAR range: [{SAR_data.min().item():.4f}, {SAR_data.max().item():.4f}]")
        print(f"  SAR mean: {SAR_data.mean().item():.4f}")
        
        # Check if data is in expected range [0, 1]
        if cloudy_data.min() < -0.1 or cloudy_data.max() > 1.1:
            print("  ⚠️  WARNING: Optical data outside expected [0,1] range!")
        if SAR_data.min() < -0.1 or SAR_data.max() > 1.1:
            print("  ⚠️  WARNING: SAR data outside expected [0,1] range!")
        
        # Check if there's actual difference between cloudy and cloudfree
        diff = torch.abs(cloudy_data - cloudfree_data).mean().item()
        print(f"  Mean difference (cloudy - cloudfree): {diff:.4f}")
        if diff < 0.01:
            print("  ⚠️  WARNING: Very small difference - clouds might be minimal!")
        
        print()

if __name__ == "__main__":
    check_data_stats()
