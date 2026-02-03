
"""
batch_test_images.py

Batch processes all images listed in /kaggle/working/test.csv.
For each image:
1.  Creates a directory: "Test Images/<ImageName>"
2.  Runs the GLF-CR+ model inference (using test_image.py)
3.  Saves the Cloud-Free output, RGB visualization, and Metrics in that folder.

Usage:
    # For Reference/Ours Model
    python codes/batch_test_images.py --csv_path /kaggle/working/test.csv \
                                      --checkpoint /kaggle/input/glf-cr-plus/checkpoints/best_model.pth \
                                      --output_root "Test Images" \
                                      --model_type cross

    # For Base Model
    python codes/batch_test_images.py --csv_path /kaggle/working/test.csv \
                                      --checkpoint /kaggle/input/base-model/checkpoints/base.pth \
                                      --output_root "Test Images_Base" \
                                      --model_type base
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

# Add current directory to path to import test_image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test_image import test_single_image
except ImportError:
    print("Error: Could not import test_single_image from test_image.py")
    print("Make sure this script is in the same folder as test_image.py")
    sys.exit(1)

# Dataset Roots (Hardcoded to match generate_kaggle_data_csv.py)
WINTER_ROOT = Path("/kaggle/input/sen12ms-cr-winter")
SPRING_ROOT = Path("/kaggle/input/t-glf-cr-winter")

def get_root_by_type(dataset_type):
    if 'winter' in dataset_type.lower():
        return WINTER_ROOT
    elif 'spring' in dataset_type.lower():
        return SPRING_ROOT
    # Fallback/Default
    return WINTER_ROOT

def process_batch(csv_path, checkpoint_path, output_root, limit=None, device='cuda', model_type='cross'):
    print(f"Reading CSV: {csv_path}")
    print(f"Model Type: {model_type}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Create root output directory
    os.makedirs(output_root, exist_ok=True)
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            rows.append(row)
            
    print(f"Found {len(rows)} images in test set.")
    
    if limit:
        print(f"Limiting to first {limit} images.")
        rows = rows[:limit]

    success_count = 0
    fail_count = 0

    # CSV Format from generate_kaggle_data_csv.py:
    # 0: split_id
    # 1: dataset_type
    # 2: s1_folder
    # 3: s2_folder (CloudFree)
    # 4: s2_cloudy_folder (Optical)
    # 5: s2_filename (CloudFree GT)
    # 6: s1_filename (SAR)
    # 7: s2_cloudy_filename (Cloudy Input)

    for i, row in enumerate(tqdm(rows, desc="Processing Images")):
        try:
            split_id = row[0]
            dataset_type = row[1]
            s1_folder = row[2]
            s2_folder = row[3]
            s2_cloudy_folder = row[4]
            s2_gt_filename = row[5]
            s1_filename = row[6]
            s2_cloudy_filename = row[7]
            
            # Determine Root
            root_path = get_root_by_type(dataset_type)
            
            # Construct Absolute Inputs
            optical_path = root_path / s2_cloudy_folder / s2_cloudy_filename
            sar_path = root_path / s1_folder / s1_filename
            gt_path = root_path / s2_folder / s2_gt_filename
            
            # Image Name (remove extension)
            image_name = s2_cloudy_filename.replace('.tif', '').replace('.TIF', '')
            
            # Create Specific Output Folder
            # e.g. Test Images/ROIs2017_winter_s2_cloudy_12_p30/
            img_output_dir = os.path.join(output_root, image_name)
            
            test_single_image(
                image_path=str(optical_path),
                model_checkpoint=checkpoint_path,
                output_dir=img_output_dir,
                sar_path=str(sar_path),
                cloudfree_path=str(gt_path), # Provide GT so metrics are calculated!
                device=device,
                model_type=model_type # Passed from argument
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\n!!! Failed to process row {i}: {e}")
            fail_count += 1
            continue

    print("\n" + "="*50)
    print("Batch Processing Complete")
    print(f"Successfully Processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output Location: {output_root}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/kaggle/working/test.csv', help='Path to test.csv')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--output_root', type=str, default='Test Images', help='Root folder for outputs')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for testing')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default='cross', choices=['cross', 'base'], 
                        help='Model type: "cross" (Ours) or "base" (Baseline)')
    
    args = parser.parse_args()
    
    process_batch(args.csv_path, args.checkpoint, args.output_root, args.limit, args.device, args.model_type)
