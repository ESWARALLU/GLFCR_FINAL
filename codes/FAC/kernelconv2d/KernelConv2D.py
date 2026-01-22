# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
import torch
from torch import nn
from torch.autograd import Function
import random

# Try to import CUDA extension, fallback to standard conv if not available
try:
    import kernelconv2d_cuda
    KERNELCONV2D_AVAILABLE = True
except ImportError:
    KERNELCONV2D_AVAILABLE = False
    print("Warning: kernelconv2d_cuda not available. Using standard convolution fallback.")


class KernelConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_size):
        # This function is ONLY called when CUDA extension is available
        ctx.kernel_size = kernel_size
        assert (input.is_contiguous() == True)
        assert (kernel.is_contiguous() == True)
        ctx.save_for_backward(input, kernel)
        assert (ctx.kernel_size == int((kernel.size(1) / input.size(1)) ** 0.5))
        intKernelSize = ctx.kernel_size
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intOutputHeight = kernel.size(2)
        intOutputWidth = kernel.size(3)

        assert (intInputHeight - intKernelSize == intOutputHeight - 1)
        assert (intInputWidth - intKernelSize == intOutputWidth - 1)

        # Updated for modern PyTorch: use device() instead of deprecated device_of()
        device = input.device
        with torch.cuda.device(device):
            output = input.new_zeros(intBatches, intInputDepth, intOutputHeight, intOutputWidth)
            kernelconv2d_cuda.forward(input, kernel, intKernelSize, output)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        intKernelSize = ctx.kernel_size
        grad_output = grad_output.contiguous()
        device = input.device
        
        grad_input = input.new_zeros(input.size())
        grad_kernel = kernel.new_zeros(kernel.size())
        with torch.cuda.device(device):
            kernelconv2d_cuda.backward(input, kernel, intKernelSize, grad_output, grad_input, grad_kernel)

        return grad_input, grad_kernel, None


def _fallback_kernel_conv2d(input, kernel, kernel_size):
    """Fallback implementation using standard PyTorch convolution when CUDA extension is unavailable."""
    B, C, H_in, W_in = input.shape
    H_out = H_in - kernel_size + 1
    W_out = W_in - kernel_size + 1
    output = torch.zeros(B, C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Unfold input to get patches
    patches = torch.nn.functional.unfold(input, kernel_size=kernel_size)  # (B, C*K*K, H_out*W_out)
    patches = patches.view(B, C, kernel_size * kernel_size, H_out, W_out)  # (B, C, K*K, H_out, W_out)
    
    # Reshape kernel for element-wise multiplication
    kernel_reshaped = kernel.view(B, C, kernel_size, kernel_size, H_out, W_out)  # (B, C, K, K, H_out, W_out)
    
    # Compute output by summing over spatial kernel dimensions
    # Use the reshaped kernel (B, C, K, K, H_out, W_out)
    for i in range(kernel_size):
        for j in range(kernel_size):
            k_slice = kernel_reshaped[:, :, i, j, :, :]  # (B, C, H_out, W_out)
            p_slice = patches[:, :, i * kernel_size + j, :, :]  # (B, C, H_out, W_out)
            output += p_slice * k_slice
    
    return output


def _fallback_kernel_conv2d_autograd(input, kernel, kernel_size):
    """
    Fallback implementation that supports gradients without CUDA extension.
    Simple element-wise operations that PyTorch can auto-differentiate efficiently.
    """
    B, C, H_in, W_in = input.shape
    H_out = H_in - kernel_size + 1
    W_out = W_in - kernel_size + 1
    
    # Unfold input patches: (B, C*K*K, H_out*W_out)
    patches = torch.nn.functional.unfold(input, kernel_size=kernel_size)
    # Reshape to (B, C, K*K, H_out*W_out) then (B, C, K, K, H_out, W_out)
    patches_reshaped = patches.view(B, C, kernel_size * kernel_size, H_out * W_out)
    patches_reshaped = patches_reshaped.view(B, C, kernel_size, kernel_size, H_out, W_out)
    
    # Kernel shape: (B, C*K*K, H_out, W_out)
    # Reshape to (B, C, K, K, H_out, W_out) for element-wise multiplication
    kernel_view = kernel.view(B, C, kernel_size, kernel_size, H_out, W_out)
    
    # Multiply patches by kernel and sum over (C, K, K) to get output (B, H_out, W_out)
    # But we want output shape (B, C, H_out, W_out)
    # Actually, let's multiply and sum over just (K, K) first, keeping (B, C, H_out, W_out)
    output = (patches_reshaped * kernel_view).sum(dim=(2, 3))  # Sum over K, K -> (B, C, H_out, W_out)
    
    return output


def gradient_check():
    kernel_size_list = [1, 3]
    len_list = [8, 10]
    for i in range(10):
        B = random.randint(1, 4)
        C = i + 1
        K = random.choice(kernel_size_list)
        H = random.choice(len_list)
        W = random.choice(len_list)
        input = torch.randn(B, C, H + K - 1, W + K - 1, requires_grad=True).cuda()
        kernel = torch.randn(B, C * K * K, H, W, requires_grad=True).cuda()
        # linear function, thus eps set to 1e-1
        print(torch.autograd.gradcheck(KernelConv2DFunction(K), (input, kernel), eps=1e-1, atol=1e-5, rtol=1e-3,
                                       raise_exception=True))


class KernelConv2D(nn.Module):
    def __init__(self, kernel_size):
        super(KernelConv2D, self).__init__()
        assert (kernel_size % 2 == 1)
        self.kernel_size = kernel_size
        self.pad = torch.nn.ReplicationPad2d(
            [(kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2])

    def forward(self, input, kernel):
        input_pad = self.pad(input)
        # Use CUDA extension if available, otherwise use PyTorch autograd fallback
        if KERNELCONV2D_AVAILABLE and input_pad.is_cuda:
            return KernelConv2DFunction.apply(input_pad, kernel, self.kernel_size)
        else:
            # Use PyTorch autograd for gradient computation
            return _fallback_kernel_conv2d_autograd(input_pad, kernel, self.kernel_size)
