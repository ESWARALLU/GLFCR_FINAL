"""
Safely replace data.csv with data_final.csv
"""
import os
import shutil
from pathlib import Path

data_dir = Path("../data")
old_csv = data_dir / "data.csv"
new_csv = data_dir / "data_final.csv"
backup_csv = data_dir / "data_old_backup.csv"

print("Replacing data.csv with data_final.csv...")

try:
    # Backup old file
    if old_csv.exists():
        shutil.copy2(old_csv, backup_csv)
        print(f"✓ Backed up old data.csv to {backup_csv}")
    
    # Copy new file
    shutil.copy2(new_csv, old_csv)
    print(f"✓ Copied data_final.csv to data.csv")
    
    # Verify
    with open(old_csv, 'r') as f:
        lines = f.readlines()
    
    print(f"\n✓ Verification:")
    print(f"  Total entries: {len(lines)}")
    print(f"  First line columns: {len(lines[0].split(','))}")
    
    if len(lines[0].split(',')) == 7:
        print(f"  ✓ Format correct (7 columns)")
    else:
        print(f"  ✗ Warning: Expected 7 columns, got {len(lines[0].split(','))}")
    
    print(f"\n✓ data.csv successfully updated!")
    print(f"\nYou can now run:")
    print(f"  python test_CR.py --model_name 'full_dataset_test' --notes 'Testing with complete dataset'")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print(f"\nManual steps:")
    print(f"1. Close any programs using data.csv")
    print(f"2. Delete data.csv manually")
    print(f"3. Rename data_final.csv to data.csv")
