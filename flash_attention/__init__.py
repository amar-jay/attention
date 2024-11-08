import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
min_flash_attn = load(
    name='min_flash_attn',
    sources=['main.cpp', 'flash_attention.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)
