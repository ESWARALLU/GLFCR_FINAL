"""
Test the updated dataloader with new CSV format
"""
import sys
import argparse
from dataloader import AlignedDataset, get_train_val_test_filelists

# Test with data_final.csv
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='../data') 
parser.add_argument('--data_list_filepath', type=str, default='../data/data_final.csv')
parser.add_argument('--is_test', type=bool, default=True)
parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
parser.add_argument('--cloud_threshold', type=float, default=0.2)
opts = parser.parse_args()

print("Loading file lists...")
train_list, val_list, test_list = get_train_val_test_filelists(opts.data_list_filepath)

print(f"\nDataset statistics:")
print(f"  Train: {len(train_list)} images")
print(f"  Val:   {len(val_list)} images")
print(f"  Test:  {len(test_list)} images")
print(f"  Total: {len(train_list) + len(val_list) + len(test_list)} images")

print(f"\nSample test entry (first):")
print(f"  {test_list[0]}")
print(f"  Length: {len(test_list[0])} columns")

print(f"\nTesting dataloader with first test image...")
test_dataset = AlignedDataset(opts, test_list[:1])

try:
    sample = test_dataset[0]
    print(f"\n✓ Dataloader works!")
    print(f"  Cloudy shape:    {sample['cloudy_data'].shape}")
    print(f"  Cloudfree shape: {sample['cloudfree_data'].shape}")
    print(f"  SAR shape:       {sample['SAR_data'].shape}")
    print(f"  Filename:        {sample['file_name']}")
    print(f"\n  Cloudy range:    [{sample['cloudy_data'].min():.4f}, {sample['cloudy_data'].max():.4f}]")
    print(f"  Cloudfree range: [{sample['cloudfree_data'].min():.4f}, {sample['cloudfree_data'].max():.4f}]")
    print(f"  SAR range:       [{sample['SAR_data'].min():.4f}, {sample['SAR_data'].max():.4f}]")
    print(f"\n✓ All checks passed! Dataset is ready.")
except Exception as e:
    print(f"\n✗ Error loading data:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
