import numpy as np
import rasterio
import pandas as pd
import random
from pathlib import Path

# Read CSV
csv_path = Path("../data/data.csv")
df = pd.read_csv(csv_path, header=None)

# Get test set (split_id = 3)
test_df = df[df[0] == 3]
print(f"Test set size: {len(test_df)}")

# Sample 20 random test images
sample_indices = random.sample(range(len(test_df)), min(20, len(test_df)))

cloud_diffs = []
for idx in sample_indices:
    row = test_df.iloc[idx]
    
    # Build paths
    data_folder = Path("../data")
    s2_path = data_folder / row[2] / row[4]
    s2_cloudy_path = data_folder / row[3] / row[6]
    
    try:
        # Read images
        with rasterio.open(s2_path) as src:
            s2_clean = src.read()
        
        with rasterio.open(s2_cloudy_path) as src:
            s2_cloudy = src.read()
        
        # Calculate difference (measure of cloud contamination)
        diff = np.mean(np.abs(s2_clean - s2_cloudy))
        cloud_diffs.append(diff)
        
        print(f"{row[4]}: Avg pixel diff = {diff:.4f}")
    
    except Exception as e:
        print(f"Error reading {row[4]}: {e}")

if cloud_diffs:
    print(f"\n=== Cloud Contamination Analysis ===")
    print(f"Mean difference: {np.mean(cloud_diffs):.4f}")
    print(f"Std difference: {np.std(cloud_diffs):.4f}")
    print(f"Min difference: {np.min(cloud_diffs):.4f}")
    print(f"Max difference: {np.max(cloud_diffs):.4f}")
    print(f"\nNote: Higher values = more cloud contamination")
    print(f"Low values (<100) suggest minimal clouds = easier task = higher PSNR")
