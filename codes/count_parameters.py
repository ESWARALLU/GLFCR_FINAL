"""
Script to count total parameters in CloudRemovalCrossAttention model
and compare with baseline models
"""

import torch
import torch.nn as nn
from net_CR_CrossAttention import CloudRemovalCrossAttention
from net_CR_RDN import ModelCRNet  # Original RDN model


def count_parameters(model):
    """Count total trainable parameters in a model"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total += num_params
    return total


def count_parameters_by_module(model):
    """Count parameters by module for detailed breakdown"""
    module_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            total = 0
            for param in module.parameters():
                if param.requires_grad:
                    total += param.numel()
            if total > 0:
                module_params[name] = total
    return module_params


def format_params(num):
    """Format parameter count with commas and M/K suffix"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


if __name__ == "__main__":
    print("="*70)
    print("CloudRemovalCrossAttention - Parameter Count Analysis")
    print("="*70)
    
    # Create CrossAttention model
    model = CloudRemovalCrossAttention()
    total_params = count_parameters(model)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Parameters: {format_params(total_params)}")
    
    # Get detailed breakdown
    print("\n" + "="*70)
    print("Parameter Breakdown by Component")
    print("="*70)
    
    component_params = {}
    
    # Optical Encoder
    opt_enc_params = count_parameters(model.optical_encoder)
    component_params['OpticalEncoder (13→256)'] = opt_enc_params
    
    # SAR Encoder
    sar_enc_params = count_parameters(model.sar_encoder)
    component_params['SAREncoder (2→256)'] = sar_enc_params
    
    # Cross-Attention
    cross_attn_params = count_parameters(model.cross_attn)
    component_params['GlobalCrossAttention'] = cross_attn_params
    
    # Speckle Gating
    speckle_params = count_parameters(model.speckle_gating)
    component_params['SpeckleAwareGatingModule'] = speckle_params
    
    # Refinement
    refine_params = count_parameters(model.refinement)
    component_params['RefinementBlock'] = refine_params
    
    # Decoder
    decoder_params = count_parameters(model.decoder)
    component_params['Decoder'] = decoder_params
    
    # Print breakdown
    for component, params in component_params.items():
        percentage = (params / total_params) * 100
        print(f"{component:.<35} {params:>10,} ({percentage:>5.1f}%)")
    
    print("-" * 70)
    print(f"{'TOTAL':.<35} {total_params:>10,} (100.0%)")
    
    # Memory estimation
    print("\n" + "="*70)
    print("Memory Estimation (fp32)")
    print("="*70)
    
    # Parameters memory (4 bytes per float32)
    param_memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"Model Parameters:     {param_memory_mb:.2f} MB")
    
    # Activation memory (rough estimate for 128x128 input)
    # OpticalEncoder: 13 + 64 + 64 + 128 + 128 + 256 = ~643 channels
    # Estimate ~2-3x parameter memory for activations
    activation_memory_mb = param_memory_mb * 2.5
    print(f"Activations (~2.5x):  {activation_memory_mb:.2f} MB")
    
    # Optimizer states (assume Adam: 2x for momentum and variance)
    optimizer_memory_mb = param_memory_mb * 2
    print(f"Optimizer State:      {optimizer_memory_mb:.2f} MB")
    
    total_memory_mb = param_memory_mb + activation_memory_mb + optimizer_memory_mb
    print(f"{'─' * 50}")
    print(f"Total Estimated:      {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
    
    # Comparison with baseline (RDN)
    print("\n" + "="*70)
    print("Comparison with Original GLF-CR (RDN)")
    print("="*70)
    
    try:
        # Note: This requires ModelCRNet to be importable
        # If it fails, we'll show estimated comparison
        rdn_model = ModelCRNet(input_channels=15, output_channels=13)  # Adjust based on actual input
        rdn_params = count_parameters(rdn_model)
        
        print(f"GLF-CR (RDN):             {rdn_params:>12,} params")
        print(f"CrossAttention:           {total_params:>12,} params")
        print(f"Reduction:                {((rdn_params - total_params)/rdn_params * 100):>11.1f}%")
        
    except Exception as e:
        print(f"Note: Could not load RDN model for comparison ({e})")
        print("\nEstimated comparison (based on literature):")
        print(f"GLF-CR (RDN):             ~2,500,000 params (estimated)")
        print(f"CrossAttention:           {total_params:>12,} params")
        print(f"Reduction:                {((2500000 - total_params)/2500000 * 100):>11.1f}%")
    
    # Model summary
    print("\n" + "="*70)
    print("Architecture Summary")
    print("="*70)
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            optical_test = torch.randn(1, 13, 256, 256)
            sar_test = torch.randn(1, 2, 256, 256)
            output = model(optical_test, sar_test)
        
        print(f"✓ Input (Optical):  {tuple(optical_test.shape)}")
        print(f"✓ Input (SAR):      {tuple(sar_test.shape)}")
        print(f"✓ Output:           {tuple(output.shape)}")
        print(f"✓ Forward pass successful!")
        
        # FLOPs estimation (rough)
        print("\n" + "="*70)
        print("Computational Complexity (Rough Estimate)")
        print("="*70)
        
        # For 256x256 input
        encoder_flops = 2.5e9  # Encoders: ~2.5B FLOPs
        cross_attn_flops = 0.5e9  # Cross-attention: ~0.5B FLOPs
        decoder_flops = 2e9  # Decoder: ~2B FLOPs
        total_flops = encoder_flops + cross_attn_flops + decoder_flops
        
        print(f"Encoders:           {encoder_flops/1e9:.1f}B FLOPs")
        print(f"Cross-Attention:    {cross_attn_flops/1e9:.1f}B FLOPs")
        print(f"Decoder:            {decoder_flops/1e9:.1f}B FLOPs")
        print(f"{'─' * 50}")
        print(f"Total (256x256):    {total_flops/1e9:.1f}B FLOPs")
        print(f"\nNote: FLOPs vary with spatial dimensions")
        print(f"      At 512x512: ~4x FLOPs (~20B)")
        print(f"      At 128x128: ~1/4 FLOPs (~1.25B)")
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
    
    print("\n" + "="*70)
