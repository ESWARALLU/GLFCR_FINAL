"""
Generate data.csv file for the ROIs2017_winter dataset
Maps S1 (SAR), S2 (cloudfree), and S2_cloudy files
"""
import os
import csv
from pathlib import Path

# Base paths
data_root = Path("../data")
s1_base = data_root / "ROIs2017_winter_s1" / "ROIs2017_winter_s1"
s2_base = data_root / "ROIs2017_winter_s2" / "ROIs2017_winter_s2"
s2_cloudy_base = data_root / "ROIs2017_winter_s2_cloudy" / "ROIs2017_winter_s2_cloudy"

# Collect all files
print("Collecting files...")
all_files = []

# Iterate through s2 folders (cloudfree - ground truth)
for s2_folder in sorted(s2_base.iterdir()):
    if not s2_folder.is_dir():
        continue
    
    folder_name = s2_folder.name  # e.g., "s2_102"
    folder_id = folder_name.split('_')[1]  # e.g., "102"
    
    # Corresponding folders
    s1_folder = s1_base / f"s1_{folder_id}"
    s2_cloudy_folder = s2_cloudy_base / f"s2_cloudy_{folder_id}"
    
    # Check folders exist
    if not s1_folder.exists() or not s2_cloudy_folder.exists():
        print(f"Warning: Missing folders for {folder_id}")
        continue
    
    # Get all .tif files in s2 folder
    s2_files = sorted(s2_folder.glob("*.tif"))
    
    for s2_file in s2_files:
        # Extract the patch number (e.g., "p100" from "ROIs2017_winter_s2_102_p100.tif")
        filename = s2_file.name
        parts = filename.split('_')
        patch = parts[-1]  # e.g., "p100.tif"
        
        # Construct corresponding filenames
        s1_filename = f"ROIs2017_winter_s1_{folder_id}_{patch}"
        s2_cloudy_filename = f"ROIs2017_winter_s2_cloudy_{folder_id}_{patch}"
        
        # Verify files exist
        s1_file = s1_folder / s1_filename
        s2_cloudy_file = s2_cloudy_folder / s2_cloudy_filename
        
        if s1_file.exists() and s2_cloudy_file.exists():
            # Store with common identifier for matching (use s1 filename as reference)
            all_files.append({
                's1_folder': f"ROIs2017_winter_s1/ROIs2017_winter_s1/s1_{folder_id}",
                's2_folder': f"ROIs2017_winter_s2/ROIs2017_winter_s2/s2_{folder_id}",
                's2_cloudy_folder': f"ROIs2017_winter_s2_cloudy/ROIs2017_winter_s2_cloudy/s2_cloudy_{folder_id}",
                's2_filename': filename,  # Keep s2 as reference since dataloader uses fileID[4]
                'patch_id': patch
            })

print(f"Total files found: {len(all_files)}")

# Split: 70% train, 15% val, 15% test
import random
random.seed(42)  # For reproducibility
random.shuffle(all_files)

total = len(all_files)
train_size = int(0.70 * total)
val_size = int(0.15 * total)

train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Write to CSV with new format
# Format: [split_id, s1_folder, s2_folder, s2_cloudy_folder, s2_filename, s1_filename, s2_cloudy_filename]
csv_path = data_root / "data_new.csv"
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write training files (split_id = 1)
    for f in train_files:
        # Extract folder_id and patch from s2_filename
        parts = f['s2_filename'].replace('.tif', '').split('_')
        folder_id = parts[3]  # e.g., "102"
        patch = f['patch_id']  # e.g., "p100.tif"
        
        s1_filename = f"ROIs2017_winter_s1_{folder_id}_{patch}"
        s2_cloudy_filename = f"ROIs2017_winter_s2_cloudy_{folder_id}_{patch}"
        
        writer.writerow(['1', f['s1_folder'], f['s2_folder'], f['s2_cloudy_folder'], 
                        f['s2_filename'], s1_filename, s2_cloudy_filename])
    
    # Write validation files (split_id = 2)
    for f in val_files:
        parts = f['s2_filename'].replace('.tif', '').split('_')
        folder_id = parts[3]
        patch = f['patch_id']
        
        s1_filename = f"ROIs2017_winter_s1_{folder_id}_{patch}"
        s2_cloudy_filename = f"ROIs2017_winter_s2_cloudy_{folder_id}_{patch}"
        
        writer.writerow(['2', f['s1_folder'], f['s2_folder'], f['s2_cloudy_folder'], 
                        f['s2_filename'], s1_filename, s2_cloudy_filename])
    
    # Write test files (split_id = 3)
    for f in test_files:
        parts = f['s2_filename'].replace('.tif', '').split('_')
        folder_id = parts[3]
        patch = f['patch_id']
        
        s1_filename = f"ROIs2017_winter_s1_{folder_id}_{patch}"
        s2_cloudy_filename = f"ROIs2017_winter_s2_cloudy_{folder_id}_{patch}"
        
        writer.writerow(['3', f['s1_folder'], f['s2_folder'], f['s2_cloudy_folder'], 
                        f['s2_filename'], s1_filename, s2_cloudy_filename])

print(f"\ndata.csv generated successfully at: {csv_path}")
print("\nFirst 5 entries:")
with open(csv_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"  {line.strip()}")
