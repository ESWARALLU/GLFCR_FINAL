"""
Batch test all images from test.csv using the base GLF-CR (RDN) model.
Creates organized folders for each image with predicted cloud-free outputs.

Usage: python test_batch_from_csv.py --csv_path /kaggle/working/test.csv --model_checkpoint <model.pth> --output_base_dir ./Test_Images
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add codes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net_CR_RDN import RDN_residual_CR

# -----------------
# Utility functions
# -----------------

def load_tiff_image(image_path):
    """Load TIFF image and ensure proper shape (C, H, W)"""
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c <= 20 and h > c and w > c:
            image = np.transpose(image, (2, 0, 1))
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype("float32")


def normalize_optical_image(image, scale=10000):
    """Normalize optical image by dividing by scale"""
    return image / scale


def normalize_sar_image(image):
    """Normalize SAR image using standard clipping ranges"""
    clip_min = [-25.0, -32.5]
    clip_max = [0.0, 0.0]
    normalized = np.zeros_like(image)
    for channel in range(len(image)):
        data = image[channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        data -= clip_min[channel]
        normalized[channel] = data / (clip_max[channel] - clip_min[channel])
    return normalized


def load_base_model(checkpoint_path, device, crop_size=256):
    """Load the base RDN model from checkpoint"""
    model = RDN_residual_CR(crop_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("network", checkpoint)
    
    # Remove 'module.' prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Drop incompatible buffer keys
    state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def read_test_csv(csv_path):
    """
    Read test.csv and extract file information.
    Returns list of dictionaries with file paths and metadata.
    """
    test_files = []
    
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue
            
            # Parse based on CSV format
            # Format: [split_id, dataset_type, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
            if len(row) >= 7:
                file_info = {
                    'split_id': row[0],
                    's1_folder': row[1] if len(row) == 7 else row[2],
                    's2_folder': row[2] if len(row) == 7 else row[3],
                    's2_cloudy_folder': row[3] if len(row) == 7 else row[4],
                    's2_filename': row[4] if len(row) == 7 else row[5],
                    's1_filename': row[5] if len(row) == 7 else row[6],
                    's2_cloudy_filename': row[6] if len(row) == 7 else row[7],
                    'dataset_type': 'winter' if len(row) == 7 else row[1]
                }
                test_files.append(file_info)
    
    return test_files


def create_rgb_visualization(output_np, output_path):
    """Create and save RGB visualization of the output"""
    if output_np.shape[0] >= 4:
        # True color (Sentinel-2: Red=B4, Green=B3, Blue=B2)
        rgb = np.stack([output_np[3], output_np[2], output_np[1]], axis=0)
        
        # Fixed true-color stretch with mild white balance and gamma
        rgb = rgb / 10000.0  # scale to [0,1]
        rgb = np.clip(rgb, 0.0, 0.35) / 0.35  # focus on typical reflectance range
        wb_gains = np.array([1.02, 1.0, 1.10], dtype=np.float32).reshape(3, 1, 1)
        rgb = rgb * wb_gains
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1/1.4)  # gentle gamma
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
        
        rgb_pil = np.transpose(rgb, (1, 2, 0))
        plt.figure(figsize=(12, 10))
        plt.imshow(rgb_pil)
        plt.title("Cloud-Removed Image (RGB) - Base Model")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return True
    return False


def process_single_image(file_info, model, device, input_data_folder, output_base_dir):
    """
    Process a single image and save outputs to organized folder structure.
    
    Args:
        file_info: Dictionary containing file paths and metadata
        model: Loaded model
        device: CUDA or CPU device
        input_data_folder: Base folder containing input data (for winter dataset)
        output_base_dir: Base directory for all test outputs
    """
    # Extract image name (without extension)
    image_name = Path(file_info['s2_filename']).stem
    
    # Create output folder for this image
    image_output_dir = os.path.join(output_base_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Determine correct root folder based on dataset type
    dataset_type = file_info.get('dataset_type', 'winter')
    if dataset_type == 'spring':
        root_folder = '/kaggle/input/t-glf-cr-winter'
    elif dataset_type == 'fall':
        root_folder = '/kaggle/input/t-glf-cr-fall'
    else:  # winter or default
        root_folder = input_data_folder
    
    # Construct full paths with correct root
    sar_path = os.path.join(root_folder, file_info['s1_folder'], file_info['s1_filename'])
    optical_path = os.path.join(root_folder, file_info['s2_cloudy_folder'], file_info['s2_cloudy_filename'])
    
    # Verify files exist
    if not os.path.exists(sar_path):
        print(f"  ⚠ SAR file not found: {sar_path}")
        return False
    if not os.path.exists(optical_path):
        print(f"  ⚠ Optical file not found: {optical_path}")
        return False
    
    # Load images
    sar_data = load_tiff_image(sar_path)
    optical_data = load_tiff_image(optical_path)
    
    # Normalize
    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)
    
    # Convert to tensors
    sar_tensor = torch.from_numpy(sar_normalized).unsqueeze(0).to(device)
    optical_tensor = torch.from_numpy(optical_normalized).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(optical_tensor, sar_tensor)
    
    # Post-process output
    output_np = output.cpu().squeeze(0).numpy()  # (13, H, W)
    output_np = np.clip(output_np * 10000.0, 0, 10000).astype("float32")
    
    # Save only RGB visualization (PNG) - skipping TIFF to save space
    output_png_path = os.path.join(image_output_dir, f"{image_name}_predicted_rgb.png")
    create_rgb_visualization(output_np, output_png_path)
    
    return True


# -----------------
# Main batch processing
# -----------------

def batch_test_from_csv(csv_path, model_checkpoint, output_base_dir, input_data_folder, device="cuda", crop_size=256):
    """
    Process all images from test.csv and save organized outputs.
    
    Args:
        csv_path: Path to test.csv
        model_checkpoint: Path to model checkpoint
        output_base_dir: Base directory for all outputs (Test Images folder)
        input_data_folder: Root folder containing input data
        device: 'cuda' or 'cpu'
        crop_size: Crop size for the model
    """
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Read CSV
    print("=" * 60)
    print(f"Reading test CSV: {csv_path}")
    test_files = read_test_csv(csv_path)
    print(f"Found {len(test_files)} images to process")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_checkpoint}")
    model = load_base_model(model_checkpoint, device, crop_size=crop_size)
    print("✓ Model loaded successfully")
    
    # Process each image
    print("\n" + "=" * 60)
    print("Processing images...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for idx, file_info in enumerate(tqdm(test_files, desc="Processing", ncols=80)):
        image_name = Path(file_info['s2_filename']).stem
        print(f"\n[{idx+1}/{len(test_files)}] Processing: {image_name}")
        
        try:
            success = process_single_image(
                file_info, 
                model, 
                device, 
                input_data_folder, 
                output_base_dir
            )
            if success:
                successful += 1
                print(f"  ✓ Saved to: {os.path.join(output_base_dir, image_name)}")
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images: {len(test_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput directory: {output_base_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch test all images from test.csv using base GLF-CR model")
    parser.add_argument("--csv_path", type=str, default="/kaggle/working/test.csv", 
                        help="Path to test.csv file")
    parser.add_argument("--model_checkpoint", type=str, required=True, 
                        help="Path to base model checkpoint (.pth)")
    parser.add_argument("--output_base_dir", type=str, default="/kaggle/working/Test_Images", 
                        help="Base directory to save all test outputs")
    parser.add_argument("--input_data_folder", type=str, default="/kaggle/input/sen12ms-cr-winter",
                        help="Root folder containing input data")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="Device to run on")
    parser.add_argument("--crop_size", type=int, default=256, 
                        help="Crop size used for the base model (RDN)")
    args = parser.parse_args()
    
    # Verify CSV exists
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    # Verify checkpoint exists
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    
    # Check CUDA availability
    device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    batch_test_from_csv(
        csv_path=args.csv_path,
        model_checkpoint=args.model_checkpoint,
        output_base_dir=args.output_base_dir,
        input_data_folder=args.input_data_folder,
        device=device,
        crop_size=args.crop_size
    )


if __name__ == "__main__":
    main()
