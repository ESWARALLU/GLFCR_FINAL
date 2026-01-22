import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']  # Updated to C++14 for modern PyTorch

nvcc_args = [
    # Allow Visual Studio 2022 and suppress STL version warnings
    '-allow-unsupported-compiler',
    '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
    
    # Legacy architectures (Pascal, Volta, Turing)
    '-gencode', 'arch=compute_60,code=sm_60',  # Pascal (GTX 10 series)
    '-gencode', 'arch=compute_61,code=sm_61',  # Pascal (GTX 10 series)
    '-gencode', 'arch=compute_70,code=sm_70',  # Volta (V100)
    '-gencode', 'arch=compute_75,code=sm_75',  # Turing (RTX 20 series)
    
    # Modern architectures (Ampere, Ada Lovelace, Hopper)
    '-gencode', 'arch=compute_80,code=sm_80',  # Ampere (A100, RTX 30 series)
    '-gencode', 'arch=compute_86,code=sm_86',  # Ampere (RTX 30 series)
    '-gencode', 'arch=compute_89,code=sm_89',  # Ada Lovelace (RTX 40 series)
    '-gencode', 'arch=compute_90,code=sm_90',  # Hopper (H100)
    
    # Enable lineinfo for better debugging
    '--use_fast_math'
]

setup(
    version='1.0.0',
    name='kernelconv2d_cuda',
    ext_modules=[
        CUDAExtension('kernelconv2d_cuda', [
            'KernelConv2D_cuda.cpp',
            'KernelConv2D_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
