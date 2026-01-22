"""
Kaggle Checkpoint Download Helper
==================================

This script automatically handles checkpoint management in Kaggle notebooks.
After training completes, it copies all files to /kaggle/output for easy download.

Usage in Kaggle Notebook:
    1. Run your training: !python codes/train_CR_kaggle.py --batch_sz 8 ...
    2. Then run this cell to finalize downloads
    3. Go to "Output" tab to download files

Latest Kaggle API compatible (2024+)
"""

import os
import shutil
from datetime import datetime


def setup_kaggle_download(
    checkpoint_dir='/kaggle/working/checkpoints',
    output_dir='/kaggle/output',
    experiment_name='training',
    verbose=True
):
    """
    Automatically copy checkpoint files to Kaggle output directory.
    
    Args:
        checkpoint_dir: Source directory with trained checkpoints
        output_dir: Kaggle output directory (for downloading)
        experiment_name: Name of experiment (for log files)
        verbose: Print detailed output
    
    Returns:
        dict: Status of copied files
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("🔄 KAGGLE CHECKPOINT DOWNLOAD SETUP")
        print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    status = {
        'best_model': False,
        'checkpoints': False,
        'logs': False,
        'errors': []
    }
    
    # ====================================
    # 1. Copy best model checkpoint
    # ====================================
    try:
        best_model_src = os.path.join(checkpoint_dir, 'best_model.pth')
        best_model_dst = os.path.join(output_dir, 'best_model.pth')
        
        if os.path.exists(best_model_src):
            shutil.copy2(best_model_src, best_model_dst)
            size_mb = os.path.getsize(best_model_dst) / (1024 * 1024)
            if verbose:
                print(f"✅ Best model: {best_model_dst}")
                print(f"   Size: {size_mb:.2f} MB")
            status['best_model'] = True
        else:
            if verbose:
                print(f"⚠️  Best model not found: {best_model_src}")
            status['errors'].append(f"Best model not found: {best_model_src}")
    
    except Exception as e:
        if verbose:
            print(f"❌ Error copying best model: {e}")
        status['errors'].append(f"Best model copy failed: {str(e)}")
    
    # ====================================
    # 2. Copy all epoch checkpoints
    # ====================================
    try:
        checkpoints_output = os.path.join(output_dir, 'checkpoints')
        
        if os.path.exists(checkpoint_dir):
            # Remove old checkpoints output if exists
            if os.path.exists(checkpoints_output):
                shutil.rmtree(checkpoints_output)
            
            # Copy entire checkpoints directory
            shutil.copytree(checkpoint_dir, checkpoints_output)
            
            # Count files
            num_files = sum([len(files) for r, d, files in os.walk(checkpoints_output)])
            total_size_mb = sum([
                os.path.getsize(os.path.join(dirpath, filename)) / (1024 * 1024)
                for dirpath, dirnames, filenames in os.walk(checkpoints_output)
                for filename in filenames
            ])
            
            if verbose:
                print(f"✅ Checkpoints directory: {checkpoints_output}")
                print(f"   Files: {num_files}, Total size: {total_size_mb:.2f} MB")
            status['checkpoints'] = True
        else:
            if verbose:
                print(f"⚠️  Checkpoints directory not found: {checkpoint_dir}")
            status['errors'].append(f"Checkpoints dir not found: {checkpoint_dir}")
    
    except Exception as e:
        if verbose:
            print(f"❌ Error copying checkpoints: {e}")
        status['errors'].append(f"Checkpoints copy failed: {str(e)}")
    
    # ====================================
    # 3. Copy training logs
    # ====================================
    try:
        logs_dir = os.path.join(checkpoint_dir, 'logs')
        
        if os.path.exists(logs_dir):
            logs_output = os.path.join(output_dir, 'logs')
            if os.path.exists(logs_output):
                shutil.rmtree(logs_output)
            
            shutil.copytree(logs_dir, logs_output)
            
            num_logs = len(os.listdir(logs_output))
            if verbose:
                print(f"✅ Training logs: {logs_output}")
                print(f"   Log files: {num_logs}")
            status['logs'] = True
        else:
            if verbose:
                print(f"⚠️  Logs directory not found: {logs_dir}")
    
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning copying logs: {e}")
        # Don't fail completely if logs are missing
    
    # ====================================
    # Print download instructions
    # ====================================
    if verbose:
        print(f"\n{'='*70}")
        print("📥 DOWNLOAD INSTRUCTIONS")
        print(f"{'='*70}\n")
        print("✅ Files are ready for download!")
        print("\n📍 KAGGLE NOTEBOOK:")
        print("   1. Click the 'Output' tab (right side of notebook)")
        print("   2. You will see:")
        print("      • best_model.pth (best epoch checkpoint)")
        print("      • checkpoints/ (all epoch checkpoints)")
        print("      • logs/ (training logs)")
        print("   3. Click download icon next to each file")
        print("\n💻 COMMAND LINE (if using Kaggle CLI):")
        print("   kaggle kernels output <kernel-id> -p ./downloaded_checkpoint")
        print("\n🔗 GITHUB REPO:")
        print("   Repository: https://github.com/ESWARALLU/glfCRPLUS")
        print(f"\n{'='*70}\n")
        
        # Summary
        if status['errors']:
            print("⚠️  WARNINGS:")
            for error in status['errors']:
                print(f"   • {error}")
        else:
            print("✅ All files copied successfully!")
        
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
    
    return status


# ============================================================================
# USAGE IN KAGGLE NOTEBOOK
# ============================================================================
"""
Paste this in a Kaggle notebook cell AFTER training completes:

    from kaggle_checkpoint_helper import setup_kaggle_download
    
    # Setup downloads (this will copy files to /kaggle/output)
    status = setup_kaggle_download(
        checkpoint_dir='/kaggle/working/checkpoints',
        output_dir='/kaggle/output',
        experiment_name='my_training',
        verbose=True
    )
    
    # Check status
    if status['best_model']:
        print("✅ Ready to download from Kaggle Output tab!")
"""


if __name__ == '__main__':
    # For testing/standalone usage
    setup_kaggle_download(
        checkpoint_dir='/kaggle/working/checkpoints',
        output_dir='/kaggle/output',
        experiment_name='training',
        verbose=True
    )
