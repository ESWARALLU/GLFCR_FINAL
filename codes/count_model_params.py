import sys
sys.path.append('.')

import torch
from net_CR_CrossAttention import CloudRemovalCrossAttention

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_parameters_by_module(model):
    """Count parameters by major module"""
    module_params = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    
    return module_params

if __name__ == "__main__":
    # Create model
    model = CloudRemovalCrossAttention()
    
    # Count total parameters
    total, trainable = count_parameters(model)
    
    print("=" * 60)
    print("GLF-CR CrossAttention Model - Parameter Count")
    print("=" * 60)
    print(f"\nTotal parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Model size:            {total / 1e6:.2f}M parameters")
    print(f"Memory (FP32):         ~{total * 4 / 1e6:.2f} MB")
    print(f"Memory (FP16):         ~{total * 2 / 1e6:.2f} MB")
    
    # Breakdown by module
    print("\n" + "=" * 60)
    print("Parameter Breakdown by Module")
    print("=" * 60)
    
    module_params = count_parameters_by_module(model)
    for name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:.<30} {params:>12,} ({params/total*100:>5.1f}%)")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    batch_size = 2
    height, width = 256, 256
    
    optical_img = torch.randn(batch_size, 13, height, width)
    sar_img = torch.randn(batch_size, 2, height, width)
    
    with torch.no_grad():
        output = model(optical_img, sar_img)
    
    print(f"Input optical shape:   {optical_img.shape}")
    print(f"Input SAR shape:       {sar_img.shape}")
    print(f"Output shape:          {output.shape}")
    print(f"Output mode:           Direct prediction (fixed architecture)")
    
    print("\n" + "=" * 60)
    print("✓ Model initialized successfully!")
    print("=" * 60)
