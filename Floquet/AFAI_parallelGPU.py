## This .py file aims to implement codes by using GPU parallel computing to accelerate the calculation.
## 1. Multi GPU parallel computing when available
## 2. Multi CPU parallel computing
## 3. Batch processing for several functions

import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from torch import nn


class tb_floquet_tbc_cuda(nn.Module):
    def __init__(self, period, lattice_constant, J_coe, ny, nx=2, device='cuda'):
        super(tb_floquet_tbc_cuda, self).__init__()
        self.T = period
        self.nx = nx
        self.ny = ny
        self.a = lattice_constant
        self.J_coe = J_coe
        self.delta_AB = np.pi / (2 * self.T)
        self.H_disorder_cached = None  # Initialize H_disorder_cached as None
        self.device = torch.device(device)
