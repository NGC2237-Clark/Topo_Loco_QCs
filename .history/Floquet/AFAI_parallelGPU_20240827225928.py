## This .py file aims to implement codes by using GPU parallel computing to accelerate the calculation.
## 1. Multi GPU parallel computing when available
## 2. Multi CPU parallel computing
## 3. Batch processing for several functions: vectorization of various parameters

import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
# from multiprocessing import Pool
from torch import nn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import permutations
import gc
import math
import concurrent.futures
from scipy.optimize import linear_sum_assignment
import time
multiprocessing.set_start_method('spawn', force=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource
from matplotlib.colors import LinearSegmentedColormap


class tb_floquet_pbc_cuda(nn.Module): # Tight-binding model of square lattice with Floquet driving and periodic boundary conditions
    def __init__(self, period, lattice_constant, J_coe, num_y, num_x = 2, device=None):
        super(tb_floquet_pbc_cuda, self).__init__()
        self.T = period
        # self.num_cells_y = num_cells_y
        self.ny = num_y # number of sites along the y direction
        self.nx = num_x # number of sites along the x direction
        self.a = lattice_constant # Distance between adjacent site A and A between adjacent two unit cells
        self.aa = self.a / 2  # Distance between adjacent site A and B in one unit cell
        # self.aa = self.a * np.sqrt(2)/ 2  # Distance between adjacent site A and B in one unit cell
        self.J_coe = J_coe / self.T # hopping strengh
        self.delta_AB = np.pi/(2* self.T)
        # Check if device is manually set or based on GPU availability
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.sigma_plus = torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble, device=self.device)
        self.sigma_minus = torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble, device=self.device)
        # Move the model to the designated device
        self.to(self.device)
        
        # Check if multiple GPUs are available and wrap the model with DataParallel
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self = nn.DataParallel(self)

    def Hamiltonian_pbc1(self, ky, pbc='y'):
        """The time-independent Hamiltonian H1 for t < T/5 with periodic boundary conditions in either x, y, or both x and y directions"""
        if isinstance(ky, (int, float)):
            ky = torch.tensor([ky], device=self.device)
        elif not isinstance(ky, torch.Tensor):
            ky = torch.tensor(ky, device=self.device)
        else:
            ky = ky.to(self.device)
        is_batch = ky.dim() > 1 or (ky.dim() == 1 and ky.shape[0] > 1)
        batch_size = ky.shape[0] if is_batch else 1
        ky = ky.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H1 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        if self.nx % 2 == 1:  # odd nx
            for i in range(size):
                a = 2 * i
                b = self.nx + 2 * i
                if b < size:
                    H1[:, a, b] = -J_coe_tensor
                    H1[:, b, a] = -J_coe_tensor.conj()
        else:  # Even nx
            based_pairs = torch.zeros((self.nx, 2), device=self.device)
            based_pairs[0] = torch.tensor([0, self.nx], device=self.device)
            for j in range(1, self.nx):
                increment = 3 if j == self.nx // 2 else 2
                based_pairs[j] = based_pairs[j-1] + increment

            for a, b in based_pairs:
                while a < size and b < size:
                    H1[:, int(a), int(b)] = -J_coe_tensor
                    H1[:, int(b), int(a)] = -J_coe_tensor.conj()
                    a += 2 * self.nx
                    b += 2 * self.nx
        
        # For the periodic boundary in the y direction
        if pbc == 'y' or pbc == 'xy':
            phase = torch.exp(1j * ky * self.a).squeeze()
            # print(phase.shape)
            device = phase.device  # Use the device of the phase tensor
            p = 0
            while 1 + 2 * p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[:, int(a), int(b)] = -J_coe_tensor * phase
                H1[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            phase = torch.exp(1j * ky * self.aa).squeeze()
            # print(phase.shape)
            device = phase.device  # Use the device of the phase tensor

            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            # print(sigma_plus.shape)
            # print(phase)
            # Ensure phase has the correct dimensions for unsqueeze
            if phase.dim() == 0:
                phase = phase.view(1, 1, 1)
            else:
                phase = phase.unsqueeze(1).unsqueeze(2)
            
            phase = phase.expand(batch_size, 2, 2)
            # print(phase)
            # print(-J_coe_tensor)
            H1 = self.sigma_minus * (-J_coe_tensor) * phase + self.sigma_plus * (-J_coe_tensor) * phase.conj()
        return H1.squeeze() if not is_batch else H1

    def Hamiltonian_pbc2(self, kx, pbc='x'):
        '''The time-independent Hamiltonian H2 for T/5 <= t < 2T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
        if isinstance(kx, (int, float)):
            kx = torch.tensor([kx], device=self.device)
        elif not isinstance(kx, torch.Tensor):
            kx = torch.tensor(kx, device=self.device)
        else:
            kx = kx.to(self.device)
        is_batch = kx.dim() > 1 or (kx.dim() == 1 and kx.shape[0] > 1)
        batch_size = kx.shape[0] if is_batch else 1
        kx = kx.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H2 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        n = 1
        a = -1
        b = 0
        while n <= self.ny and a < size - 1 and b < size:
            a += 2
            b += 2
            if b < n * self.nx:
                H2[:, a, b] = -J_coe_tensor
                H2[:, b, a] = -J_coe_tensor.conj()
            else:
                n += 1
                if self.nx % 2 == 1 and n % 2 == 0:
                    a += 0
                    b += 0
                elif self.nx % 2 == 1 and n % 2 != 0:
                    a += 2
                    b += 2
                elif self.nx % 2 == 0:
                    a += 1
                    b += 1
                if a < size - 1 and b < size:
                    H2[:, a, b] = -J_coe_tensor
                    H2[:, b, a] = -J_coe_tensor.conj()
            if self.nx == 2:
                a += 1
                b += 1
            if b >= self.ny * self.nx - 2:
                break
        # For the periodic boundary in the x direction
        if pbc == 'x' or pbc == 'xy':
            phase = torch.exp(1j * kx * self.a).squeeze()
            device = phase.device  # Use the device of the phase tensor
            p = 0
            while self.nx - 1 + 2 * self.nx * p < size and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                H2[:, a, b] = -J_coe_tensor * phase
                H2[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            phase = torch.exp(1j * kx * self.aa).squeeze()
            device = phase.device  # Use the device of the phase tensor
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            # Ensure phase has the correct dimensions for unsqueeze
            if phase.dim() == 0:
                phase = phase.view(1, 1, 1)
            else:
                phase = phase.unsqueeze(1).unsqueeze(2)
            
            phase = phase.expand(batch_size, 2, 2)
            H2 = self.sigma_minus * (-J_coe_tensor) * phase + self.sigma_plus * (-J_coe_tensor) * phase.conj()
        return H2.squeeze() if not is_batch else H2

    def Hamiltonian_pbc3(self, ky, pbc='y'):
        '''The time-independent Hamiltonian H3 for 2T/5 <= t < 3T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
        if isinstance(ky, (int, float)):
            ky = torch.tensor([ky], device=self.device)
        elif not isinstance(ky, torch.Tensor):
            ky = torch.tensor(ky, device=self.device)
        else:
            ky = ky.to(self.device)
        is_batch = ky.dim() > 1 or (ky.dim() == 1 and ky.shape[0] > 1)
        batch_size = ky.shape[0] if is_batch else 1
        ky = ky.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H3 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        if self.nx % 2 == 1:  # odd nx
            for i in range(size):
                a = 2 * i + 1
                b = self.nx + 2 * i + 1
                if b < size:
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
        else:  # Even nx
            n = 1
            a = 1
            b = 1 + self.nx
            if b < size:
                H3[:, a, b] = -J_coe_tensor
                H3[:, b, a] = -J_coe_tensor.conj()
            while n < self.ny and a < size - 1 and b < size - 1:
                a += 2
                b += 2
                if a < n * self.nx:
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
                else:
                    n += 1
                    if n % 2 == 0:  # even n
                        a -= 1
                        b -= 1
                    elif n % 2 != 0 and b < size - 1:  # odd n
                        a += 1
                        b += 1
                    else:
                        a -= 2
                        b -= 2
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
        
        # For the periodic boundary in the y direction
        if pbc == 'y' or pbc == 'xy':
            phase = torch.exp(1j * ky * self.a).squeeze()
            device = phase.device  # Use the device of the phase tensor
            p = 0
            while 2 * p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                H3[:, int(a), int(b)] = -J_coe_tensor * phase
                H3[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            phase = torch.exp(1j * ky * self.aa).squeeze()
            device = phase.device  # Use the device of the phase tensor
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            # Ensure phase has the correct dimensions for unsqueeze
            if phase.dim() == 0:
                phase = phase.view(1, 1, 1)
            else:
                phase = phase.unsqueeze(1).unsqueeze(2)
            
            phase = phase.expand(batch_size, 2, 2)
            H3 = self.sigma_plus * (-J_coe_tensor) * phase + self.sigma_minus * (-J_coe_tensor) * phase.conj()
        return H3.squeeze(0) if not is_batch else H3

    def Hamiltonian_pbc4(self, kx, pbc='x'):
        '''The time-independent Hamiltonian H4 for 3T/5 <= t < 4T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
        if isinstance(kx, (int, float)):
            kx = torch.tensor([kx], device=self.device)
        elif not isinstance(kx, torch.Tensor):
            kx = torch.tensor(kx, device=self.device)
        else:
            kx = kx.to(self.device)
        is_batch = kx.dim() > 1 or (kx.dim() == 1 and kx.shape[0] > 1)
        # print(is_batch)
        # print(kx.dim())
        # print(kx.shape[0])
        batch_size = kx.shape[0] if is_batch else 1
        kx = kx.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H4 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        n = 1
        a = -2
        b = -1
        while n <= self.ny and a < size - 2 and b < size - 2:
            a += 2
            b += 2
            if b < n * self.nx:
                H4[:, a, b] = -J_coe_tensor
                H4[:, b, a] = -J_coe_tensor.conj()
            else:
                n += 1
                if self.nx % 2 == 0 and self.nx != 2:  # even nx
                    a += 1
                    b += 1
                elif self.nx % 2 == 0 and self.nx == 2 and b < size - 2:  # even nx and nx = 2
                    a += 2
                    b += 2
                    n += 1
                elif self.nx % 2 == 1 and n % 2 == 1:  # odd nx and n is odd
                    a += 0
                    b += 0
                elif self.nx % 2 == 1 and n % 2 == 0:  # odd nx and n is even
                    a += 2
                    b += 2
                else:
                    n += 1
                    a += -2
                    b += -2
                H4[:, a, b] = -J_coe_tensor
                H4[:, b, a] = -J_coe_tensor.conj()
        # For the periodic boundary in the x direction
        if pbc == 'x' or pbc == 'xy':
            phase = torch.exp(1j * kx * self.a).squeeze()
            device = phase.device  # Use the device of the phase tensor
            p = 0
            while 2 * self.nx * (1 + p) - 1 < size and self.nx % 2 == 0:
                a = 2 * self.nx * (1 + p) - 1
                # print(a)
                b = 2 * self.nx * p + self.nx
                # print(b)
                # print(phase)
                H4[:, a, b] = -J_coe_tensor * phase
                H4[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            phase = torch.exp(1j * kx * self.aa).squeeze()
            device = phase.device  # Use the device of the phase tensor
            H4 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            # Ensure phase has the correct dimensions for unsqueeze
            if phase.dim() == 0:
                phase = phase.view(1, 1, 1)
            else:
                phase = phase.unsqueeze(1).unsqueeze(2)
            
            phase = phase.expand(batch_size, 2, 2)
            H4 = self.sigma_plus * (-J_coe_tensor) * phase + self.sigma_minus * (-J_coe_tensor) * phase.conj()
        return H4.squeeze() if not is_batch else H4

    def Hamiltonian_pbc_onsite(self, delta=None):
        '''The time-independent Hamiltonian H5 for 4T/5 <= t < T with any boundary conditions'''
        size = self.nx * self.ny
        H_onsite = torch.zeros((size, size), dtype=torch.cdouble, device=self.device)

        if delta is None:
            delta = self.delta_AB

        deltas = torch.full((size,), delta, dtype=torch.cdouble, device=self.device)
        deltas[1::2] *= -1  # Alternate the sign starting from the second element

        # Adjust the sign based on row blocks
        for n in range(self.ny):
            if n % 2 == 1:
                start_idx = n * self.nx
                end_idx = start_idx + self.nx
                deltas[start_idx:end_idx] *= -1

        H_onsite[torch.arange(size), torch.arange(size)] = deltas
        return H_onsite

    def Hamiltonian_pbc(self, t, kx, ky, delta=None, reverse=False, pbc='x'):
        '''The time-independent Hamiltonian H(t) with periodic boundary conditions in the x direction and open boundary conditions in the y direction'''
        if delta is None:
            delta = self.delta_AB
        
        H_onsite = self.Hamiltonian_pbc_onsite(delta)
        
        # Ensure t is within [0, T)
        t = t % self.T
        
        if not reverse:  # Anti-clockwise
            if t < self.T/5:
                H = self.Hamiltonian_pbc1(ky, pbc) + H_onsite
            elif self.T/5 <= t < 2 * self.T/5:
                H = self.Hamiltonian_pbc2(kx, pbc) + H_onsite
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                H = self.Hamiltonian_pbc3(ky, pbc) + H_onsite
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                H = self.Hamiltonian_pbc4(kx, pbc) + H_onsite
            else:  # 4 * self.T/5 <= t < self.T
                H = H_onsite
        else:  # Clockwise
            if t < self.T/5:
                H = self.Hamiltonian_pbc2(kx, pbc) + H_onsite
            elif self.T/5 <= t < 2 * self.T/5:
                H = self.Hamiltonian_pbc1(ky, pbc) + H_onsite
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                H = self.Hamiltonian_pbc4(kx, pbc) + H_onsite
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                H = self.Hamiltonian_pbc3(ky, pbc) + H_onsite
            else:  # 4 * self.T/5 <= t < self.T
                H = H_onsite
        return H

    def time_evolution_operator_pbc(self, t, n, kx, ky, pbc, delta=None, reverse=False):
        '''The time evolution operator U(t) = exp(-iH(t)) with periodic boundary conditions'''
        '''The n is the order of the Taylor expansion'''
        
        def compute_U_single(kx_val, ky_val):
            H_onsite = self.Hamiltonian_pbc_onsite(delta)
            if reverse:
                H1 = self.Hamiltonian_pbc2(kx_val, pbc) + H_onsite
                H2 = self.Hamiltonian_pbc1(ky_val, pbc) + H_onsite
                H3 = self.Hamiltonian_pbc4(kx_val, pbc) + H_onsite
                H4 = self.Hamiltonian_pbc3(ky_val, pbc) + H_onsite
                H5 = H_onsite
            else:
                H1 = self.Hamiltonian_pbc1(ky_val, pbc) + H_onsite
                H2 = self.Hamiltonian_pbc2(kx_val, pbc) + H_onsite
                H3 = self.Hamiltonian_pbc3(ky_val, pbc) + H_onsite
                H4 = self.Hamiltonian_pbc4(kx_val, pbc) + H_onsite
                H5 = H_onsite

            local_n = n
            is_unitary = False
            while not is_unitary:
                if t < self.T/5:
                    U = self.taylor_expansion(H1, t, local_n)
                elif self.T/5 <= t < 2 * self.T/5:
                    U1 = self.taylor_expansion(H1, self.T/5, local_n)
                    U2 = self.taylor_expansion(H2, t - self.T/5, local_n)
                    U = U2 @ U1
                elif 2 * self.T/5 <= t < 3 * self.T/5:
                    U1 = self.taylor_expansion(H1, self.T/5, local_n)
                    U2 = self.taylor_expansion(H2, self.T/5, local_n)
                    U3 = self.taylor_expansion(H3, t - 2 * self.T/5, local_n)
                    U = U3 @ U2 @ U1
                elif 3 * self.T/5 <= t < 4 * self.T/5:
                    U1 = self.taylor_expansion(H1, self.T/5, local_n)
                    U2 = self.taylor_expansion(H2, self.T/5, local_n)
                    U3 = self.taylor_expansion(H3, self.T/5, local_n)
                    U4 = self.taylor_expansion(H4, t - 3 * self.T/5, local_n)
                    U = U4 @ U3 @ U2 @ U1
                else:  # 4 * self.T/5 <= t <= self.T
                    U1 = self.taylor_expansion(H1, self.T/5, local_n)
                    U2 = self.taylor_expansion(H2, self.T/5, local_n)
                    U3 = self.taylor_expansion(H3, self.T/5, local_n)
                    U4 = self.taylor_expansion(H4, self.T/5, local_n)
                    U5 = self.taylor_expansion(H5, t - 4 * self.T/5, local_n)
                    U = U5 @ U4 @ U3 @ U2 @ U1

                U_dagger = U.conj().transpose(-2, -1)
                product = U_dagger @ U
                identity = torch.eye(self.nx * self.ny, dtype=torch.cdouble, device=self.device)
                is_unitary = torch.allclose(product, identity, atol=1e-8)
                local_n += 1
            return U

        if pbc == 'xy':
            U = torch.stack([torch.stack([compute_U_single(kx_val, ky_val) for ky_val in ky]) for kx_val in kx])
        else:
            U = compute_U_single(kx, ky)

        return U

    def taylor_expansion(self, H, t, n):
        U = torch.zeros_like(H, dtype=torch.cdouble, device=self.device)
        for i in range(n+1):
            U += (1/math.factorial(i)) * (-1j * t) ** i * torch.matrix_power(H, i)
        return U

    # Split operator decomposition (also known as Suzuki-Trotter decomposition)
    def infinitesimal_evol_operator(self, Hr, V_dis, dt):
        '''The infinitesimal evolution operator for the real space Hamiltonian'''
        U = torch.matrix_exp(-1j * (Hr) * dt/2) @ torch.matrix_exp(-1j * (V_dis) * dt) @ torch.matrix_exp(-1j * (Hr) * dt/2)
        return U
    
    def time_evolution_operator_pbc1(self, t, steps_per_segment, kx, ky, pbc, delta=None, reverse=False):
        '''Time evolution operator for time 0 ≤ t ≤ T and even t>T with a specified number of steps per T/5 segment'''
        '''Support not only scalar (kx, ky, t) pair
        but also batch processing for multiple (kx, ky, t) pairs: vectorization of kx, ky, and t: 1D tensors
        the output shape is then (N_kx, N_ky, N_t, nx*ny, nx*ny)'''
        
        if delta is None:
            delta = self.delta_AB
        
        # Determine if inputs are batched
        is_batch = isinstance(kx, torch.Tensor) and isinstance(ky, torch.Tensor) and kx.dim() + ky.dim() > 1

        if is_batch:
            batch_x = kx.shape[0]  # Total number of (kx, ky) pairs
            batch_y = ky.shape[0]
            # print(batch_x, batch_y)
        else:
            batch_x = 1
            batch_y = 1
        
        # Handle scalar or tensor `t`
        if isinstance(t, torch.Tensor) and t.dim() > 0:
            batch_t = t.shape[0]  # Total number of time t
            t = t.view(-1)
        else:
            batch_t = 1
            t = torch.tensor([t], device=self.device, dtype=torch.float64)  # Convert scalar `t` to 1D tensor
        
        # Reshape inputs for batch processing
        if isinstance(kx, torch.Tensor):
            kx = kx.view(batch_x, 1)
        else:
            kx = torch.full((batch_x, 1), kx, device=self.device)

        if isinstance(ky, torch.Tensor):
            ky = ky.view(batch_y, 1)
        else:
            ky = torch.full((batch_y, 1), ky, device=self.device)

        t = t.view(batch_t, 1)
        
        # Calculate dt based on the number of steps per segment
        # print("steps_per_segment type:", type(steps_per_segment))
        # print("self.T type:", type(self.T))
        dt = self.T / (5 * steps_per_segment)
        # dt = dt.to(self.device)
        H_onsite = self.Hamiltonian_pbc_onsite(delta)
        # print('H_onsite', H_onsite)
        H_onsite = H_onsite.to(self.device)
        # Broadcast H_onsite to match batch size (N_t, N_kx, N_ky, nx*ny, nx*ny)
        H_onsite = H_onsite.unsqueeze(0).unsqueeze(1).unsqueeze(1).expand(batch_t, batch_x, batch_y, -1, -1)
        # print('H_onsite', H_onsite.shape)
        if not reverse:  # Anti-clockwise
            H1 = self.Hamiltonian_pbc1(ky, pbc).unsqueeze(0).expand(batch_x, -1, -1, -1)
            # print('H1', H1.shape)
            H2 = self.Hamiltonian_pbc2(kx, pbc).unsqueeze(1).expand(-1, batch_y,-1, -1)
            # print('H2', H2.shape)
            H3 = self.Hamiltonian_pbc3(ky, pbc).unsqueeze(0).expand(batch_x, -1, -1, -1)
            # print('H3', H3.shape)
            H4 = self.Hamiltonian_pbc4(kx, pbc).unsqueeze(1).expand(-1, batch_y,-1, -1)
            # print('H4', H4.shape)
        else:  # Clockwise
            H1 = self.Hamiltonian_pbc2(kx, pbc).unsqueeze(1).expand(-1, batch_y,-1, -1)
            H2 = self.Hamiltonian_pbc1(ky, pbc).unsqueeze(0).expand(batch_x, -1, -1, -1)
            H3 = self.Hamiltonian_pbc4(kx, pbc).unsqueeze(1).expand(-1, batch_y,-1, -1)
            H4 = self.Hamiltonian_pbc3(ky, pbc).unsqueeze(0).expand(batch_x, -1, -1, -1)
        ## Adding the batch dimension for different time t so that the dimensions of H are (N_t, N_kx, N_ky, nx*ny, nx*ny)
        H1 = H1.unsqueeze(0).expand(batch_t, -1, -1, -1, -1).to(self.device)
        H2 = H2.unsqueeze(0).expand(batch_t, -1, -1, -1, -1).to(self.device)
        H3 = H3.unsqueeze(0).expand(batch_t, -1, -1, -1, -1).to(self.device)
        H4 = H4.unsqueeze(0).expand(batch_t, -1, -1, -1, -1).to(self.device)
        # print("H1", H1.shape)
        identity = torch.eye(self.nx * self.ny, dtype=torch.cdouble, device=self.device).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_t, batch_x, batch_y, -1, -1)
        U = torch.eye(self.nx * self.ny, dtype=torch.cdouble, device=self.device).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_t, batch_x, batch_y, -1, -1)
        # print("U",U.shape)
        total_steps = torch.floor(t / dt).long()
        # print('total_steps',total_steps)
        max_steps = total_steps.max().item()
        for step in range(max_steps):
            current_t = step * dt
            current_t_mod = current_t % self.T
            active = step < total_steps
            # print(active.shape)
            # active = active.squeeze().unsqueeze(1).unsqueeze(2).expand(-1, batch_size, batch_size)
            active = active.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
            # print(active.shape)
            mask1 = (current_t_mod < self.T/5) &  active
            mask2 = (current_t_mod >= self.T/5) & (current_t_mod < 2*self.T/5) & active
            mask3 = (current_t_mod >= 2*self.T/5) & (current_t_mod < 3*self.T/5) & active
            mask4 = (current_t_mod >= 3*self.T/5) & (current_t_mod < 4*self.T/5) & active
            mask5 = (current_t_mod >= 4*self.T/5) & active
            # print(step, mask1, mask2, mask3, mask4, mask5)
            H0 = torch.zeros_like(H1).to(self.device)
            # print(H0.shape)
            Hr = torch.zeros_like(H1).to(self.device)
            # print(Hr.shape)
            Hr[mask1] = H1[mask1]
            Hr[mask2] = H2[mask2]
            Hr[mask3] = H3[mask3]
            Hr[mask4] = H4[mask4]
            Hr[mask5] = H0[mask5]
            combined_mask = mask1 | mask2 | mask3 | mask4 | mask5
            combined_mask = combined_mask.to(self.device)
            combined_mask = combined_mask.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, self.nx * self.ny, self.nx * self.ny)
            # print(combined_mask)
            # print(combined_mask)
            # temp_result = self.infinitesimal_evol_operator(Hr, H_onsite, dt)
            # print(f"infinitesimal_evol_operator result device: {temp_result.device}")
            # print("Device check:")
            # print(f"combined_mask device: {combined_mask.device}")
            # print(f"Hr device: {Hr.device}")
            # print(f"H_onsite device: {H_onsite.device}")
            # print(f"identity device: {identity.device}")
            U_step = torch.where(combined_mask, \
                                self.infinitesimal_evol_operator(Hr, H_onsite, dt), identity)
            U = U_step @ U
        
        # Handle any remaining time
        remaining_time = t - total_steps * dt
        # print('remaining_time',remaining_time)
        mask1 = (t < self.T/5) & (remaining_time > 0)
        mask2 = (t >= self.T/5) & (t < 2*self.T/5) & (remaining_time > 0)
        mask3 = (t >= 2*self.T/5) & (t < 3*self.T/5) & (remaining_time > 0)
        mask4 = (t >= 3*self.T/5) & (t < 4*self.T/5) & (remaining_time > 0)
        mask5 = (t >= 4*self.T/5) & (remaining_time > 0)
        
        mask1 = mask1.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
        mask2 = mask2.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
        mask3 = mask3.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
        mask4 = mask4.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
        mask5 = mask5.view(batch_t, 1, 1).expand(-1, batch_x, batch_y)
        H0 = torch.zeros_like(H1)
        Hr = torch.zeros_like(H1)
        Hr[mask1] = H1[mask1]
        Hr[mask2] = H2[mask2]
        Hr[mask3] = H3[mask3]
        Hr[mask4] = H4[mask4]
        Hr[mask5] = H0[mask5]
        combined_mask = mask1 | mask2 | mask3 | mask4 | mask5
        combined_mask = combined_mask.to(self.device)
        combined_mask = combined_mask.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, self.nx * self.ny, self.nx * self.ny)
        # print(combined_mask)
        remaining_time = remaining_time.view(batch_t, 1, 1, 1, 1).expand(-1, batch_x, batch_y, self.nx * self.ny, self.nx * self.ny)
        remaining_time = remaining_time.to(self.device)
        U_step = torch.where(combined_mask, 
                            self.infinitesimal_evol_operator(Hr, H_onsite, remaining_time), 
                            identity)
        U = U_step @ U
        U = U.permute(1, 2, 0, 3, 4)
        return U.squeeze()
    
    def quasienergy_eigenstates(self, t, k_num, steps_per_segment, delta=None, reverse=False, plot=False, save_path=None, pbc='x'):
        
        def compute_sorted_eigensystem(U):
            eigvals, eigvecs = torch.linalg.eig(U)
            # print('eigvecs shape', eigvecs.shape)
            
            # Compute E_T
            E_T = 1j * torch.log(eigvals) / t
            eigv_r = E_T.real
            # print("Original eigvecs (first point):", eigvecs[1, 1])
            # Sort the eigenvalues based on their real parts
            sorted_indices = torch.argsort(eigv_r, dim=-1)
            
            # Reorder the eigenvalues and their real parts
            sorted_E_T = torch.gather(eigvals, -1, sorted_indices) ## Sorted complex eigenvalues
            sorted_eigv_r = torch.gather(eigv_r, -1, sorted_indices)
            
            # Reorder the eigenvectors
            expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
            sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)
            # print("First sorted eigenvector (for verification):", sorted_eigvecs[1, 1])
            # print("Sorted eigenvectors shape:", sorted_eigvecs.shape)
            # print("Sorted eigenvectors norm:", torch.norm(sorted_eigvecs, dim=-2))
            return sorted_eigv_r, sorted_E_T, sorted_eigvecs
        
        def plot_quasienergies(k_values, eigenvalues, dim):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.tick_params(axis='both', labelsize=32)
            ax.set_xticks([0, torch.pi/3, 2*torch.pi/3, torch.pi, 4*torch.pi/3, 5*torch.pi/3, 2*torch.pi])
            ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
            
            k_values_np = k_values.cpu().numpy()
            eigenvalues_np = eigenvalues.cpu().numpy()
            
            for i in range(k_num):
                ax.scatter([k_values_np[i]] * eigenvalues.shape[1], eigenvalues_np[i, :], color='black', s=0.1)
            
            ax.set_xlabel(f'$k_{dim}$', fontsize=34)
            ax.set_xlim(0, 2*torch.pi/self.a)
            ax.set_ylim(-torch.pi/t, torch.pi/t)
            
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

        pbc_configs = {
            'x': (lambda: (torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64),
                        torch.zeros(1, device=self.device, dtype=torch.float64)), 'x'),
            'y': (lambda: (torch.zeros(1, device=self.device, dtype=torch.float64),
                        torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)), 'y'),
            'xy': (lambda: (torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64),
                            torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)), 'xy')
        }

        k_x, k_y = pbc_configs[pbc][0]()
        pbc_dim = pbc_configs[pbc][1]

        U = self.time_evolution_operator_pbc1(t, steps_per_segment, k_x, k_y, pbc_dim, delta, reverse)
        # print(U)
        eigenvalues_matrix, eigval, wf_matrix = compute_sorted_eigensystem(U)

        if plot:
            if pbc != 'xy':
                plot_quasienergies(k_x if pbc == 'x' else k_y, eigenvalues_matrix, pbc)
            else:
                # 3D plotting for xy case
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X, Y = torch.meshgrid(k_x.cpu(), k_y.cpu(), indexing='ij')
                X_np, Y_np = X.numpy(), Y.numpy()
                eigenvalues_matrix_np = eigenvalues_matrix.cpu().numpy()
                
                for i in range(self.nx * self.ny):
                    ax.plot_surface(X_np, Y_np, eigenvalues_matrix_np[:, :, i], cmap='viridis')
                
                ax.set_xlabel(r'$k_{x}$', fontsize=15)
                ax.set_ylabel(r'$k_{y}$', fontsize=15)
                ax.set_zlabel('Quasienergy', fontsize=15)
                ax.set_zlim(-torch.pi/t, torch.pi/t)
                ax.view_init(elev=2, azim=5)
                plt.show()

        return eigenvalues_matrix, eigval, wf_matrix
    
    def quasienergy_eigenstates0(self, t, k_num, n, delta=None, reverse=False, plot=False, save_path=None, pbc='x'):
        
        def compute_sorted_eigensystem(U):
            eigvals, eigvecs = torch.linalg.eig(U)
            # print('eigvecs shape', eigvecs.shape)
            # Compute E_T
            E_T = 1j * torch.log(eigvals) / t
            eigv_r = E_T.real
            # print("Original eigvecs (first point):", eigvecs[1, 1])
            # Sort the eigenvalues based on their real parts
            sorted_indices = torch.argsort(eigv_r, dim=-1)
            
            # Reorder the eigenvalues and their real parts
            sorted_E_T = torch.gather(E_T, -1, sorted_indices)
            sorted_eigv_r = torch.gather(eigv_r, -1, sorted_indices)
            
            # Reorder the eigenvectors
            expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
            sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)
            # print("First sorted eigenvector (for verification):", sorted_eigvecs[1, 1])
            # print("Sorted eigenvectors shape:", sorted_eigvecs.shape)
            # print("Sorted eigenvectors norm:", torch.norm(sorted_eigvecs, dim=-2))
            return sorted_eigv_r, sorted_eigvecs
        
        def plot_quasienergies(k_values, eigenvalues, dim):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.tick_params(axis='both', labelsize=32)
            ax.set_xticks([0, torch.pi/3, 2*torch.pi/3, torch.pi, 4*torch.pi/3, 5*torch.pi/3, 2*torch.pi])
            ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
            
            k_values_np = k_values.cpu().numpy()
            eigenvalues_np = eigenvalues.cpu().numpy()
            
            for i in range(k_num):
                ax.scatter([k_values_np[i]] * eigenvalues.shape[1], eigenvalues_np[i, :], color='black', s=0.1)
            
            ax.set_xlabel(f'$k_{dim}$', fontsize=34)
            ax.set_xlim(0, 2*torch.pi/self.a)
            ax.set_ylim(-torch.pi/t, torch.pi/t)
            
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

        pbc_configs = {
            'x': (lambda: (torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64),
                        torch.zeros(1, device=self.device, dtype=torch.float64)), 'x'),
            'y': (lambda: (torch.zeros(1, device=self.device, dtype=torch.float64),
                        torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)), 'y'),
            'xy': (lambda: (torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64),
                            torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)), 'xy')
        }

        k_x, k_y = pbc_configs[pbc][0]()
        pbc_dim = pbc_configs[pbc][1]

        U = self.time_evolution_operator_pbc(t, n, k_x, k_y, pbc_dim, delta, reverse)
        eigenvalues_matrix, wf_matrix = compute_sorted_eigensystem(U)

        if plot:
            if pbc != 'xy':
                plot_quasienergies(k_x if pbc == 'x' else k_y, eigenvalues_matrix, pbc)
            else:
                # 3D plotting for xy case
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X, Y = torch.meshgrid(k_x.cpu(), k_y.cpu(), indexing='ij')
                X_np, Y_np = X.numpy(), Y.numpy()
                eigenvalues_matrix_np = eigenvalues_matrix.cpu().numpy()
                
                for i in range(self.nx * self.ny):
                    ax.plot_surface(X_np, Y_np, eigenvalues_matrix_np[:, :, i], cmap='viridis')
                
                ax.set_xlabel(r'$k_{x}$', fontsize=15)
                ax.set_ylabel(r'$k_{y}$', fontsize=15)
                ax.set_zlabel('Quasienergy', fontsize=15)
                ax.set_zlim(-torch.pi/t, torch.pi/t)
                ax.view_init(elev=2, azim=5)
                plt.show()

        return eigenvalues_matrix, wf_matrix
    
    ## The following functions calculate the winding numbers of the quasienergy spectrum starting from obtaining the eigenvalues and eigenvectors on the grid of the quasienergy spectrum
    def eigen_grid(self, ini, N_div, steps_per_segment, delta=None, reverse=False, plot=False, save_path=None, pbc='xy'):
        '''Version 1
        The eigenvalues and eigenvectors on the grid of the quasienergy spectrum'''
        k_x = torch.linspace(ini/self.a, (ini+2*torch.pi)/self.a, N_div+1, device=self.device)
        k_y = torch.linspace(ini/self.a, (ini+2*torch.pi)/self.a, N_div+1, device=self.device)
        t = torch.linspace(0, self.T, N_div+1, device=self.device)
        ## The eigenvalues and eigenvectors on the grid of the quasienergy spectrum
        ## The dimension of the eigenvalues_matrix is (N_kx, N_ky, N_t, nx*ny)
        ## The dimension of the wf_matrix is (N_kx, N_ky, N_t, nx*ny, nx*ny)
        U_tensor = self.time_evolution_operator_pbc1(t, steps_per_segment, k_x, k_y, pbc, delta, reverse)
        eigvals, eigvecs = torch.linalg.eig(U_tensor)
        # print("Original eigvecs (first point):", eigvecs[-1, -1, -1])
        # print("Original eigvecs norm:", torch.norm(eigvecs, dim=-2))
        # Compute the real part of the eigenvalues for sorting
        eigv = -1j * torch.log(eigvals)
        # print(eigv)
        eigv_r = eigv.real
        
        # Sort the eigenvalues based on their real parts
        sorted_indices = torch.argsort(eigv_r, dim=-1)
        
        # Reorder the eigenvalues, their real parts of the log(eigenvalues), and the eigenvectors
        sorted_eigvals = torch.gather(eigvals, -1, sorted_indices)
        sorted_eigv_r = torch.gather(eigv_r, -1, sorted_indices)
        
        expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
        sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)
        # print("First sorted eigenvector (for verification):", sorted_eigvecs[-1, -1, -1])
        # print("Sorted eigenvectors shape:", sorted_eigvecs.shape)
        # print("Sorted eigenvectors norm:", torch.norm(sorted_eigvecs, dim=-2))
        if plot == True:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
            def update(frame):
                ax.clear()
                ax.set_ylim(0, 1.1)
                
                # Get eigenvalues for this time frame
                eigvals_t = sorted_eigvals[:, :, frame, :].reshape(-1)
                
                # Debugging prints
                # print(f"Frame {frame}: Min abs: {torch.min(torch.abs(eigvals_t))}, Max abs: {torch.max(torch.abs(eigvals_t))}")
                # print(f"Number of eigenvalues: {eigvals_t.numel()}")
                
                # Convert to numpy and handle potential complex numbers
                eigvals_np = eigvals_t.cpu().numpy()
                angles = np.angle(eigvals_np)
                magnitudes = np.abs(eigvals_np)
                
                # Plot eigenvalues on complex unit circle
                scatter = ax.scatter(angles, magnitudes, alpha=0.5, s=30)
                
                ax.set_title(f'Eigenvalues at t = {t[frame].item():.2f}')
                return scatter,
            
            ani = animation.FuncAnimation(fig, update, frames=N_div+1, blit=True)
            
            if save_path is not None:
                ani.save(save_path, writer='ffmpeg')
            plt.show()
        return sorted_eigv_r, sorted_eigvals, sorted_eigvecs
        
    def animate_quasienergy_spectra(self, N_div, steps_per_segment, delta=None, reverse=False, fps=5, filename= None):
        sorted_eigv_r, _, _ = self.eigen_grid(0, N_div, steps_per_segment, delta=delta, reverse=reverse, pbc='xy')
        k_x = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device).cpu().numpy()
        k_y = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device).cpu().numpy()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        def update(frame):
            ax1.clear()
            kx, ky = np.meshgrid(k_x, k_y)
            z = sorted_eigv_r[:, :, frame].cpu().numpy()
            for i in range(z.shape[-1]):
                ax1.plot_surface(kx, ky, z[:, :, i], cmap='viridis')
            ax1.set_xlabel('k_x')
            ax1.set_ylabel('k_y')
            ax1.set_zlabel('Quasienergy')
            ax1.set_title(f'Time Shot: {frame}')
            ax1.set_zlim(-torch.pi, torch.pi)
            ax1.view_init(elev=2, azim=5)
        ani = animation.FuncAnimation(fig, update, frames=sorted_eigv_r.shape[2], interval=1000/fps, blit=False)
        
        if filename is not None:
            ani.save(filename, writer='ffmpeg')

        plt.show()
    
    def animate_combined_spectra(self, N_div, steps_per_segment, ini=0, delta=None, reverse=False, fps=5, filename= None):
        sorted_eigv_r, sorted_eigvals, _ = self.eigen_grid(ini, N_div, steps_per_segment, delta=delta, reverse=reverse, pbc="xy")
        k_x = torch.linspace(ini, (2*torch.pi+ini)/self.a, N_div+1, device=self.device).cpu().numpy()
        k_y = torch.linspace(ini, (2*torch.pi+ini)/self.a, N_div+1, device=self.device).cpu().numpy()
        t = torch.linspace(0, self.T, N_div+1, device=self.device)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='polar')
        def update(frame):
            ax1.clear()
            ax2.clear()
            # 3D quasienergy plot
            kx, ky = np.meshgrid(k_x, k_y)
            z = sorted_eigv_r[:, :, frame].cpu().numpy()
            for i in range(z.shape[-1]):
                ax1.plot_surface(kx, ky, z[:, :, i], cmap='viridis')
            ax1.set_xlabel(r'$k_{x}$')
            ax1.set_ylabel(r'$k_{y}$')
            ax1.set_zlabel(r'$T\epsilon$')
            ax1.set_title(f'Time Shot: {frame}')
            ax1.set_zlim(-torch.pi, torch.pi)
            ax1.view_init(elev=2, azim=5)
            # Polar plot of eigenvalues
            eigvals_t = sorted_eigvals[:, :, frame, :].reshape(-1)
            # Convert to numpy and handle potential complex numbers
            eigvals_np = eigvals_t.cpu().numpy()
            angles = np.angle(eigvals_np)
            magnitudes = np.abs(eigvals_np)
            # Plot eigenvalues on complex unit circle
            scatter = ax2.scatter(angles, magnitudes, alpha=0.5, s=30)
            ax2.set_ylim(0, 1.1)
            ax2.set_yticks([1])  # Only show tick at r=1
            ax2.set_title(f'Eigenvalues at t = {t[frame].item():.2f}')
            
            return ax1, scatter
        
        ani = animation.FuncAnimation(fig, update, frames=sorted_eigv_r.shape[2], interval=1000/fps, blit=False)
        
        if filename is not None:
            ani.save(filename, writer='ffmpeg')
        plt.show()
    
    def sort_eigensystem(self, eigvals, eigvalss, eigvecs):
        kx, ky, t, size = eigvals.shape
        device = eigvals.device

        # Initialize arrays to store sorted indices and results
        sorted_indices = torch.zeros((kx, ky, t, size), dtype=torch.long, device=device)
        sorted_eigvals = torch.zeros_like(eigvals)
        sorted_eigvalss = torch.zeros_like(eigvalss)
        sorted_eigvecs = torch.zeros_like(eigvecs)

        # Sort for each (kx, t) point along ky direction
        for i in range(kx):
            for k in range(t):
                # Initialize sorting for ky=0
                sorted_indices[i, 0, k] = torch.arange(size, device=device)
                sorted_eigvals[i, 0, k] = eigvals[i, 0, k]
                sorted_eigvalss[i, 0, k] = eigvalss[i, 0, k]
                sorted_eigvecs[i, 0, k] = eigvecs[i, 0, k]

                # Sort for ky > 0
                for j in range(1, ky):
                    # Compute overlap between current and previous eigenvectors
                    overlap = torch.abs(torch.matmul(
                        eigvecs[i, j, k].conj().transpose(-2, -1),
                        sorted_eigvecs[i, j-1, k]
                    ))

                    # Use linear_sum_assignment to find the best matching
                    cost_matrix = -overlap.cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # Store the sorted indices
                    sorted_indices[i, j, k] = torch.tensor(col_ind, dtype=torch.long, device=device)

                    # Apply sorting to eigvals, eigvalss, and eigvecs
                    sorted_eigvals[i, j, k] = eigvals[i, j, k, col_ind]
                    sorted_eigvalss[i, j, k] = eigvalss[i, j, k, col_ind]
                    sorted_eigvecs[i, j, k] = eigvecs[i, j, k, :, col_ind]

        return sorted_eigvals, sorted_eigvalss, sorted_eigvecs
    
    def U_nu(self, s_nu_pi, s_nu_pj):
        """Version 1: Support batch processing: Checked
        Compute U_nu as defined in the paper.

        Parameters:
        s_nu_pi (torch.Tensor): Eigenvector at point pi
        s_nu_pj (torch.Tensor): Eigenvector at point pj

        Returns:
        torch.Tensor: U_nu value
        """
        inner_product = torch.sum(s_nu_pi.conj() * s_nu_pj, dim=0)
        # print(inner_product)
        abs_inner_product = torch.abs(inner_product)
        # print(abs_inner_product)
        result = inner_product / abs_inner_product
        return result
    
    def mod(self, a, b):
        return ((a - 1) % b) + 1

    def face_F_hat(self, i, j, k, N_div, eigvals, eigvecs):
        """Version 2: Batch processing of three alpha values 
        Compute F̂νp,α for all three faces (α = 1, 2, 3) simultaneously.
    
        Parameters:
        i, j, k (int): Indices of the base point of the faces
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        N_div (int): Number of divisions in each dimension
        Returns:
        torch.Tensor: F̂νp,α values for all three faces, shape (3, num_bands)
        """
        # Define the vertices of the faces for all three alphas
        ## Version 1
        # vertices = torch.tensor([
        #     # alpha = 1
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, (k+1)%N_div], 
        #     [i%N_div, j%N_div, (k+1)%N_div]],
        #     # alpha = 2
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [i%N_div, j%N_div, (k+1)%N_div], 
        #     [(i+1)%N_div, j%N_div, (k+1)%N_div], 
        #     [(i+1)%N_div, j%N_div, k%N_div]],
        #     # alpha = 3
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [(i+1)%N_div, j%N_div, k%N_div], 
        #     [(i+1)%N_div, (j+1)%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, k%N_div]]
        # ], dtype=torch.long, device=self.device)
        ## Version 2
        vertices = torch.tensor([
            # alpha = 1
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k+1, N_div)], 
            [self.mod(i, N_div), self.mod(j, N_div), self.mod(k+1, N_div)]],
            # alpha = 2
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j, N_div), self.mod(k+1, N_div)], 
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k+1, N_div)], 
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k, N_div)]],
            # alpha = 3
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)],
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i+1, N_div), self.mod(j+1, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k, N_div)]]
        ], dtype=torch.long, device=self.device)
        
        # print("vertices for face", vertices)
        # Get eigenvectors at each vertex for all faces
        v = [[eigvecs[tuple(v)] for v in face] for face in vertices]
        # Compute U_nu for each edge of each face
        U12 = torch.stack([self.U_nu(v[alpha][0], v[alpha][1]) for alpha in range(3)])
        U23 = torch.stack([self.U_nu(v[alpha][1], v[alpha][2]) for alpha in range(3)])
        U34 = torch.stack([self.U_nu(v[alpha][2], v[alpha][3]) for alpha in range(3)])
        U41 = torch.stack([self.U_nu(v[alpha][3], v[alpha][0]) for alpha in range(3)])
        
        # Compute F̂νp,α
        F_hat = (-1 / (2 * torch.pi * 1j)) * torch.log(U12 * U23 * U34 * U41)
        return F_hat.real
    
    def cube(self, i, j, k, N_div, eigvals, eigvecs):
        """
        Compute the cube function as defined in equation 4.3 of the paper for all bands simultaneously.
        
        Parameters:
        i, j, k (int): Indices of the base point of the cube
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        N_div (int): Number of divisions in each dimension
        
        Returns:
        torch.Tensor: Cube function values for each band
        """
        F_p = self.face_F_hat(i, j, k, N_div, eigvals, eigvecs)
        
        C_p = torch.zeros(F_p.shape[1], dtype=torch.float64, device=self.device)
        
        for alpha in range(3):
            i_plus, j_plus, k_plus = i, j, k
            if alpha == 0:
                i_plus = (i + 1)
            elif alpha == 1:
                j_plus = (j + 1)
            else:  # alpha == 2
                k_plus = (k + 1)
            
            F_p_plus = self.face_F_hat(i_plus, j_plus, k_plus, N_div, eigvals, eigvecs)
            
            C_p += F_p_plus[alpha] - F_p[alpha] # Modified (Deviated) from the algorithm in the paper
        
        C_p = torch.round(C_p)
        
        # if torch.sum(C_p) != 0:
        #     print(f"Warning: The sum of the elements in C_p is non-zero at cube ({i}, {j}, {k}).")
        
        return C_p
    
    def determine_m(self, i, j, k, eigvals):
        """Version 2
        Determine m^nu_p,alpha for all alpha and all bands simultaneously
        
        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid        
        Returns:
        torch.Tensor: m^nu_p,alpha for all alphas and all bands, shape (3, num_bands)
        """
        # Get eigenvalues at p
        phi_p = eigvals[i, j, k]
        
        # Calculate indices for p + δ_alpha for all three directions
        indices_plus = torch.tensor([
            [(i + 1), j, k],
            [i, (j + 1), k],
            [i, j, (k + 1)]], dtype=torch.long, device=self.device)
            
        # Get eigenvalues at p + δ_alpha for all directions
        phi_p_plus_delta = eigvals[indices_plus[:, 0], indices_plus[:, 1], indices_plus[:, 2]]

        # Calculate the difference for all directions
        diff = (phi_p - phi_p_plus_delta)
        
        # Determine m for all alphas and all bands
        m = - torch.floor((diff + torch.pi) / (2 * torch.pi))
        # print(m)
        return m.long()
        
    def determine_M(self, i, j, k, eigvals, C_p):
        """Version 2
        Determine M^ν_p when Ĉ^ν_p ≠ 0 for two indices ν, ν'
        
        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        C_p (torch.Tensor): Ĉ^ν_p values
        
        Returns:
        torch.Tensor: M^ν_p values for all bands
        """
        ## Get eigenvalues at p
        # Get eigenvalues at p
        phi_p = eigvals[i, j, k]
        
        M_p = torch.zeros_like(C_p)
        non_zero_indices = torch.nonzero(C_p).squeeze()
        
        if len(non_zero_indices) == 2:
            nu, nu_prime = non_zero_indices
            diff = phi_p[nu] - phi_p[nu_prime]
            # print(diff)
            M_nu = - torch.floor((diff + torch.pi) / (2 * torch.pi))
            M_p[nu] = M_nu
        
        return M_p
        
    def w3(self, ini, N_div, steps_per_segment, delta=None, reverse=False):
        eigvals, eigvalss, eigvecs = self.eigen_grid(ini, N_div, steps_per_segment, pbc="xy")
        # eigvals, eigvalss, eigvecs = self.sort_eigensystem(eigvals, eigvalss, eigvecs)
        # print(eigvals.shape)
        n_band = self.nx * self.ny
        w3 = 0
        delta = 1/N_div
        delta_space = delta * torch.pi * 2 / self.a
        delta_t = delta * self.T
        for i in range(N_div):
            for j in range(N_div):
                for k in range(N_div): 
                    #print(r"($i_1, i_2, i_3$)", i,j,k)
                    p = torch.tensor([delta_space * (i), delta_space * (j), delta_t * (k)], dtype=torch.float64, device=self.device)
                    # print('p', p)
                    C_p = self.cube(i, j, k, N_div, eigvals, eigvecs)
                    # print(r'$C_p$', C_p)
                    M_p = self.determine_M(i, j, k, eigvals, C_p)
                    # print(r'$M_p$', M_p)
                    F_p = self.face_F_hat(i, j, k, N_div, eigvals, eigvecs)
                    m_p = self.determine_m(i, j, k, eigvals)
                    if torch.any(M_p != 0):
                        print(f"($i_1, i_2, i_3$)", i,j,k)
                        print('p', p)
                        print(f'$C_p$', C_p)
                        print(f'$M_p$', M_p)
                        print(f'$F_p$', F_p)
                        print(f'$m_p$', m_p)
                        print("\n")
                    if torch.any(m_p != 0):
                        print(f'$F_p$', F_p)
                        print(f'$m_p$', m_p)
                        print("\n")
                    # Update W3
                    w3 += torch.sum(C_p * M_p) + torch.sum(F_p * m_p)
        return w3, eigvalss, eigvecs
    
    def log_with_branch_cut(self, z, branch_cut_angle=0):
        # Ensure the branch cut angle is a tensor
        if not isinstance(branch_cut_angle, torch.Tensor):
            branch_cut_angle = torch.tensor([branch_cut_angle], dtype=torch.float64, device=z.device)
        else:
            # Ensure branch_cut_angle is on the same device as z
            branch_cut_angle = branch_cut_angle.to(device=z.device, dtype=torch.float64)
        # Step 1: Compute the magnitude of z
        magnitude = torch.abs(z)
        # Step 2: Compute the initial phase
        initial_phase = torch.angle(z)
        
        # Normalize the branch_cut_angle to be within [-pi, pi)
        branch_cut_angle = (branch_cut_angle + torch.pi) % (2 * torch.pi) - torch.pi

        # Step 3: Adjust phase to be within [branch_cut_angle, branch_cut_angle + 2*pi)
        adjusted_phase = initial_phase.unsqueeze(-1) - branch_cut_angle
        adjusted_phase = (adjusted_phase + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi)
        adjusted_phase += branch_cut_angle  # Shift back to the desired range

        # Correct phases that are not within the specified range
        adjusted_phase += torch.where(adjusted_phase < branch_cut_angle, 2 * torch.pi, 0)
        adjusted_phase -= torch.where(adjusted_phase >= branch_cut_angle + 2 * torch.pi, 2 * torch.pi, 0)

        # Step 4: Compute the logarithm using the new angle and magnitude
        log_z = torch.log(magnitude).unsqueeze(-1) + 1j * adjusted_phase
        
        return log_z
    
    def determine_K(self, i, j, k, eigvals, eigvecs, branch_cut_angle):
        """
        Determine K^ν_(i,j) such that |-i log_ξ d^ν(p) - φ^ν_p + 2πK^ν_(i,j)| < π at i3 = N for all bands simultaneously

        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        branch_cut_angle (float): Angle for the branch cut in radians

        Returns:
        torch.Tensor: K^ν_(i,j) values for all bands
        """
        # Get eigenvalues at p
        d_p = eigvals[i, j, k]
        # Compute φ^ν_p using -i log(d^nu_p)
        phi_p = -1j * torch.log(d_p)
        # print("without branch cut",phi_p)
        # Compute -i log_ξ d^ν(p) using the provided branch cut angle
        log_xi_d = -1j * self.log_with_branch_cut(d_p, branch_cut_angle)
        # print('with branch cut', log_xi_d)
        # Compute the difference
        # Handle both scalar and tensor branch_cut_angle
        if log_xi_d.dim() > phi_p.dim():
            diff = log_xi_d - phi_p.unsqueeze(-1)
        else:
            diff = log_xi_d - phi_p
        # print("diff", diff)
        # Determine K
        K = -torch.floor((diff.real + torch.pi) / (2 * torch.pi))
        
        return K.long()
    
    def winding3(self, ini, N_div, steps_per_segment, branch_cut_angle, plot=False, delta=None, reverse=False):
        # print("winding3 arguments:")
        # print(f"N_div: {N_div}")
        # print(f"steps_per_segment: {steps_per_segment}")
        # print(f"branch_cut_angle: {branch_cut_angle}")
        # print(f"plot: {plot}")
        # print(f"delta: {delta}")
        # print(f"reverse: {reverse}")
        w3, eigvals, eigvecs = self.w3(ini, N_div, steps_per_segment, delta, reverse)
        # print(eigvals.shape)
        print(w3)
        # Initialize correction_term based on branch_cut_angle type
        if isinstance(branch_cut_angle, torch.Tensor):
            correction_term = torch.zeros(len(branch_cut_angle), device=self.device)
        else:
            correction_term = 0
        # Iterate over the 2D grid at μ3 = 1 (i3 = N_div)
        for i1 in range(N_div):
            for i2 in range(N_div):
                # Compute the base point p
                # print(r"($i_1, i_2$)", i1,i2)
                # p = torch.tensor([(i1) * 2*torch.pi/(self.a * N_div),
                #                 (i2) * 2*torch.pi/(self.a * N_div),
                #                 self.T], dtype=torch.float64, device=self.device)
                # print('p', p)
                # Compute F^ν_p,3
                vertices = torch.tensor([
                    [(i1)%N_div, (i2)%N_div, N_div],
                    [(i1+1)%N_div, (i2)%N_div, N_div],
                    [(i1+1)%N_div, (i2+1)%N_div, N_div],
                    [(i1)%N_div, (i2+1)%N_div, N_div]
                ], dtype=torch.long, device=self.device)
                # print(vertices)
                # Compute F^ν_p,3 using these vertices
                v = [eigvecs[tuple(v)] for v in vertices]
                U12 = self.U_nu(v[0], v[1])
                U23 = self.U_nu(v[1], v[2])
                U34 = self.U_nu(v[2], v[3])
                U41 = self.U_nu(v[3], v[0])
                F_p_3 = (- 1 / (2 * torch.pi * 1j)) * torch.log(U12 * U23 * U34 * U41)
                F_p_3 = F_p_3.real
                # print(F_p_3.shape)
                # Determine K^ν_(i,j)
                K = self.determine_K(i1, i2, N_div, eigvals, eigvecs, branch_cut_angle)
                # print(K.shape)
                # Compute the product and sum over bands
                # Handle both scalar and tensor branch_cut_angle
                if isinstance(branch_cut_angle, torch.Tensor):
                    term = torch.sum(F_p_3.unsqueeze(-1) * K, dim=0)
                else:
                    term = torch.sum(F_p_3 * K)
                
                # Add to the correction term
                correction_term += term
        print(correction_term)
        W3_U_xi = (w3 + correction_term)
        if plot:
            # Plot for multiple branch cut angles
            tick_label_fontsize = 32
            label_fontsize = 34
            plt.figure(figsize=(10, 6))
            plt.plot(branch_cut_angle.cpu().numpy(), W3_U_xi.cpu().numpy(), '-o')
            plt.xlabel('Branch Cut Angle, $\\varepsilon$', fontsize=label_fontsize)
            plt.ylabel('$W_{3}[U_{\\varepsilon}]$', fontsize=label_fontsize)
            plt.ylim(-1.5, 1.5)
            plt.yticks(range(-1, 2), fontsize=tick_label_fontsize)  # This sets integer ticks from -1 to 1 with specified fontsize
            plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                    [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'], fontsize=tick_label_fontsize)
            plt.grid(True)
            plt.show()
        return W3_U_xi
    
    def log_with_branchcut1(self, z, epsilonT):
        """
        Compute the logarithm with a branch cut as defined in the AFAI paper.
        
        Args:
        z (torch.Tensor): Complex tensor of eigenvalues
        epsilonT (float): Branch cut parameter times T (epsilon * T)
        
        Returns:
        torch.Tensor: Logarithm of z with the specified branch cut
        """
        # Compute chi as the phase of z
        chi = torch.angle(z)
        
        # Normalize chi to be in the range [0, 2π)
        chi = chi % (2 * torch.pi)
        
        # Adjust chi based on the branch cut definition
        chi_adjusted = torch.where(
            chi < epsilonT,
            chi,
            chi - 2 * torch.pi
        )
        
        return chi_adjusted * 1j
    
    def H_eff(self, k_num, steps_per_segment, epsilonT = torch.pi, delta=None, reverse=False, pbc='xy'):
        delete, eigenvalues_matrix, wf_matrix = self.quasienergy_eigenstates(self.T, k_num, steps_per_segment, delta, reverse, plot=False, pbc=pbc)
        del delete
        log_eigenvalues = self.log_with_branchcut1(eigenvalues_matrix, epsilonT)
        # Initialize H_eff with the same shape and device as wf_matrix
        # Multiply by (1j / self.T) here
        log_eigenvalues = (1j / self.T) * log_eigenvalues

        H_eff = torch.zeros_like(wf_matrix, dtype=torch.complex128)

        # Reshape eigenvalues_matrix to (kx*ky, size) and convert to complex
        eigenvalues_flat = log_eigenvalues.reshape(-1, log_eigenvalues.shape[-1]).to(torch.complex128)

        # Create diagonal matrices for all k-points at once
        H_diag = torch.diag_embed(eigenvalues_flat)

        # Reshape wf_matrix to (kx*ky, size, size)
        wf_flat = wf_matrix.reshape(-1, wf_matrix.shape[-2], wf_matrix.shape[-1])

        # Ensure wf_flat is complex
        wf_flat = wf_flat.to(torch.complex128)

        # Compute H_eff for all k-points in one batch operation
        H_eff_flat = torch.bmm(torch.bmm(wf_flat, H_diag), wf_flat.conj().transpose(-2, -1))

        # Reshape H_eff back to original shape
        H_eff = H_eff_flat.reshape_as(wf_matrix)

        return H_eff, log_eigenvalues, wf_matrix
    
    def get_k_values(self, k_num, pbc):
        if pbc == 'x':
            k_x = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device)
            k_y = torch.zeros(1, device=self.device)
        elif pbc == 'y':
            k_x = torch.zeros(1, device=self.device)
            k_y = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device)
        elif pbc == 'xy':
            k_x = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device)
            k_y = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device)
        return k_x, k_y

    def compute_deformed_U(self, t_num, k_num, steps_per_segment, epsilonT=torch.pi, delta=None, reverse=False, pbc='xy'):
        # Generate t values
        t = torch.linspace(0, self.T, t_num, device=self.device)
        # print('t shape', t.shape)
        # First, compute U(kx, ky, t) for 0 <= t <= T
        k_x, k_y = self.get_k_values(k_num, pbc)  # Adjusted to k_num
        U_t = self.time_evolution_operator_pbc1(t, steps_per_segment, k_x, k_y, pbc, delta, reverse)
        # print('U_t shape', U_t.shape)
        # Compute H_eff and get log_eigenvalues and S (eigenvectors)
        H_eff, log_eigenvalues, S = self.H_eff(k_num, steps_per_segment, epsilonT, delta, reverse, pbc)
        # print("log_eigenvalues shape:", log_eigenvalues.shape)
        del H_eff
        # Reshape t for broadcasting
        t_reshaped = t.reshape(1, 1, -1, 1)
        # print('t_reshaped shape', t_reshaped.shape)
        # Compute exp(1j*t*log_eigenvalues) for all time steps
        exp_term = torch.exp(1j * t_reshaped * log_eigenvalues.unsqueeze(2))
        # print('exp_term shape', exp_term.shape)
        # Reshape exp_term for matrix multiplication
        exp_term_diag = torch.diag_embed(exp_term)
        
        # Expand S to match the time dimension
        S_expanded = S.unsqueeze(2).expand(-1, -1, t_num, -1, -1)
        # print("S_expanded shape:", S_expanded.shape)
        # print("exp_term_diag shape:", exp_term_diag.shape)
        # print("S_expanded.conj().transpose(-1, -2) shape:", S_expanded.conj().transpose(-1, -2).shape)
        # Compute S * exp(1j*t*log_eigenvalues) * S^+
        deformation_factor = torch.einsum('...tij,...tjk,...tkl->...til', S_expanded, exp_term_diag, S_expanded.conj().transpose(-1, -2))
        
        # Compute the deformed U for all time steps
        U_prime = torch.einsum('...tij,...tjk->...tik', U_t, deformation_factor)
        
        return U_prime

    
    def commutator(self, A, B):
        return torch.matmul(A, B) - torch.matmul(B, A)
    
    def compute_U_derivatives_central(self, U):
        """
        Compute the derivatives of U with respect to kx, ky, and t using central differences.

        Args:
        U (torch.Tensor): The U tensor of shape (k_num+1, k_num+1, t_num+1, size, size)

        Returns:
        tuple: (dU_dkx, dU_dky, dU_dt) with shapes:
            dU_dkx: (k_num-1, k_num+1, t_num+1, size, size)
            dU_dky: (k_num+1, k_num-1, t_num+1, size, size)
            dU_dt: (k_num+1, k_num+1, t_num-1, size, size)
        """
        k_num, _, t_num, size, _ = U.shape
        k_num -= 1
        t_num -= 1

        # Compute step sizes
        dkx = (2 * torch.pi / self.a) / k_num
        # print('d_kx',dkx)
        dky = (2 * torch.pi / self.a) / k_num
        # print('d_ky',dky)
        dt = self.T / t_num
        # print('dt', dt)
        # Compute central differences for derivatives
        dU_dkx = (U[2:, :, :, :, :] - U[:-2, :, :, :, :]) / (2 * dkx)
        dU_dky = (U[:, 2:, :, :, :] - U[:, :-2, :, :, :]) / (2 * dky)
        dU_dt = (U[:, :, 2:, :, :] - U[:, :, :-2, :, :]) / (2 * dt)
        # Debug prints before trimming
        # print('Before trimming:')
        # print('shape of dU_dkx', dU_dkx.shape)  # Shape: (k_num-1, k_num+1, t_num+1, size, size)
        # print('shape of dU_dky', dU_dky.shape)  # Shape: (k_num+1, k_num-1, t_num+1, size, size)
        # print('shape of dU_dt', dU_dt.shape)    # Shape: (k_num+1, k_num+1, t_num-1, size, size)

        # Trim the boundaries to ensure consistent shape
        dU_dkx = dU_dkx[:, 1:-1, 1:-1, :, :]
        dU_dky = dU_dky[1:-1, :, 1:-1, :, :]
        dU_dt = dU_dt[1:-1, 1:-1, :, :, :]

        # Debug prints after trimming
        # print('After trimming:')
        # print('shape of dU_dkx', dU_dkx.shape)  # Expected: (k_num-1, k_num-1, t_num-1, size, size)
        # print('shape of dU_dky', dU_dky.shape)  # Expected: (k_num-1, k_num-1, t_num-1, size, size)
        # print('shape of dU_dt', dU_dt.shape)    # Expected: (k_num-1, k_num-1, t_num-1, size, size)
        
        return dU_dkx, dU_dky, dU_dt

    def compute_integrand(self, U):
        """
        Compute the integrand for the triple integral in formula 8 using central differences.

        Args:
        U (torch.Tensor): The U tensor of shape (k_num+1, k_num+1, t_num+1, size, size)

        Returns:
        torch.Tensor: The integrand of shape (k_num-2, k_num-2, t_num-2)
        """
        dU_dkx, dU_dky, dU_dt = self.compute_U_derivatives_central(U)

        # Trim U to match the shape of derivatives
        U = U[1:-1, 1:-1, 1:-1, :, :]
        U_dag = U.conj().transpose(-2, -1)
        # print('the shape of trimmed U', U.shape)
        term1 = torch.matmul(U_dag, dU_dt)
        # print('the shape of term1', term1.shape)
        term2 = self.commutator(torch.matmul(U_dag, dU_dkx), torch.matmul(U_dag, dU_dky))
        # print('the shape of term2', term2.shape)
        integrand = torch.einsum('...ii', torch.matmul(term1, term2))
        # print('integrand', integrand.shape)
        real_integrand = integrand.real
        # print(integrand[0,0])
        return real_integrand

    def compute_winding_number(self, U):
        """
        Compute the winding number W3 using the central difference method for the triple integral.

        Args:
        U (torch.Tensor): The U tensor of shape (k_num+1, k_num+1, t_num+1, size, size)

        Returns:
        float: The computed winding number.
        """
        integrand = self.compute_integrand(U)
        
        # Get the dimensions of the integrand
        k_num, _, t_num, size, _ = U.shape
        k_num -= 1
        t_num -= 1

        # Compute step sizes
        dkx = (2 * torch.pi / self.a) / k_num
        # print('d_kx',dkx)
        dky = (2 * torch.pi / self.a) / k_num
        # print('d_ky',dky)
        dt = self.T / t_num
        # print('dt', dt)

        # Apply central difference integration
        integral_sum = torch.sum(integrand)
        winding_number = (1 / (8 * torch.pi**2)) * integral_sum * dkx * dky * dt

        return winding_number
    
    def plot_W3_vs_xi(self, k_num, t_num, steps_per_segment, N_xi, delta=None, reverse=False):
        """
        Generate a plot of W₃[Uξ] versus ξ for different branch cut values.

        Args:
        k_num (int): Number of k-points in each direction.
        t_num (int): Number of time steps.
        steps_per_segment (int): Number of steps per segment in the time evolution.
        N_xi (int): Number of ξ values to compute.
        delta (float, optional): Delta parameter for time evolution.
        reverse (bool, optional): Reverse parameter for time evolution.
        pbc (str, optional): Periodic boundary conditions.

        Returns:
        None (displays the plot)
        """
        # Generate ξ values
        xi_values = torch.linspace(0, 2*torch.pi, N_xi, device=self.device)
        
        # Initialize W3_values as a zero tensor
        W3_values = torch.zeros(N_xi, device=self.device)

        for i, xi in enumerate(xi_values):
            # Compute the deformed U for this ξ value
            U_xi = self.compute_deformed_U(t_num, k_num, steps_per_segment, epsilonT=xi, delta=delta, reverse=reverse, pbc='xy')
            
            # Compute W₃[Uξ] for this ξ value and store it directly in W3_values
            W3_values[i] = self.compute_winding_number(U_xi)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(xi_values.cpu().numpy(), W3_values.cpu().numpy(), '-o')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$W_3[U_\xi]$')
        # plt.title(r'Winding Number $W_3[U_\xi]$ vs $\xi$')
        plt.grid(True)
        # plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference
        
        # Set x-axis ticks to multiples of π
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

        plt.show()

    def convergence_w3(self, grid_num_range, steps_per_segment, delta=None, reverse=False, save_path=None):
        '''Convergence test for the deformed'''
        print('grid', grid_num_range)
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        bulk_invariant_values = torch.zeros((len(grid_num_range), len(xi_values)), device=self.device)
        for i, num in enumerate(grid_num_range):
            for j, xi in enumerate(xi_values):
                print(i,j)
                print('num', num)
                U_xi = self.compute_deformed_U(t_num=num, k_num=num, steps_per_segment=steps_per_segment, epsilonT=xi, delta=delta, reverse=reverse, pbc='xy')
                bulk_invariant_values[i,j] = self.compute_winding_number(U_xi)
        # Convert to numpy arrays for plotting
        grid_np = grid_num_range.cpu().numpy() if isinstance(grid_num_range, torch.Tensor) else np.array(grid_num_range)
        bulk_np = bulk_invariant_values.cpu().numpy()
        # Font sizes
        tick_label_fontsize = 32
        label_fontsize = 34
        legend_fontsize = 32

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(grid_np, bulk_np[:, 0], label=r'$\varepsilon = 0$', marker='o')
        ax.plot(grid_np, bulk_np[:, 1], label=r'$\varepsilon = \pi$', marker='s')
        ax.set_xlabel(r'$N$', fontsize=label_fontsize)
        ax.set_ylabel(r'$W_3[U_\varepsilon]$', fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.grid(True)

        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        
        plt.show()
        return bulk_invariant_values
        
class tb_floquet_tbc_cuda(nn.Module):
    def __init__(self, period, lattice_constant, J_coe, ny, nx=2, device=None):
        super(tb_floquet_tbc_cuda, self).__init__()
        self.T = period
        self.nx = nx
        self.ny = ny
        self.a = lattice_constant
        self.J_coe = J_coe / self.T
        self.delta_AB = np.pi / (2 * self.T)
        self.H_disorder_cached = None

        # Check if device is manually set or based on GPU availability
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move the model to the designated device
        self.to(self.device)

        # Check if multiple GPUs are available and wrap the model with DataParallel
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self = nn.DataParallel(self)

    def lattice_numbering(self):
        '''Numbering the sites in the lattice'''
        return torch.arange(self.ny * self.nx, device=self.device).reshape(self.ny, self.nx)
    
    def Hamiltonian_tbc1(self, theta_y, tbc='y'):
        """The time-independent Hamiltonian H1 for t < T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space"""
        if isinstance(theta_y, (int, float)):
            theta_y = torch.tensor([theta_y], device=self.device)
        elif not isinstance(theta_y, torch.Tensor):
            theta_y = torch.tensor(theta_y, device=self.device)
        else:
            theta_y = theta_y.to(self.device)
        is_batch = theta_y.dim() > 1 or (theta_y.dim() == 1 and theta_y.shape[0] > 1)
        batch_size = theta_y.shape[0] if is_batch else 1
        theta_y = theta_y.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H1 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        if self.nx % 2 == 1:  # odd nx
            for i in range(size):
                a = 2 * i
                b = self.nx + 2 * i
                if b < size:
                    H1[:, a, b] = -J_coe_tensor
                    H1[:, b, a] = -J_coe_tensor.conj()
        else:  # Even nx
            based_pairs = torch.zeros((self.nx, 2), device=self.device)
            based_pairs[0] = torch.tensor([0, self.nx], device=self.device)
            for j in range(1, self.nx):
                if j == self.nx // 2:
                    increment = 3
                else:
                    increment = 2
                based_pairs[j] = based_pairs[j - 1] + increment

            for a, b in based_pairs:
                while a < size and b < size:
                    H1[:, int(a), int(b)] = -J_coe_tensor
                    H1[:, int(b), int(a)] = -J_coe_tensor.conj()
                    a += 2 * self.nx
                    b += 2 * self.nx
        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            phase = torch.exp(1j * theta_y).squeeze()
            while 1 + 2 * p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[:, int(a), int(b)] = -J_coe_tensor * phase
                H1[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        return H1.squeeze() if not is_batch else H1
    
    def Hamiltonian_tbc2(self, theta_x, tbc='x'):
        '''The time-independent Hamiltonian H2 for T/5 <= t < 2T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        if isinstance(theta_x, (int, float)):
            theta_x = torch.tensor([theta_x], device=self.device)
        elif not isinstance(theta_x, torch.Tensor):
            theta_x = torch.tensor(theta_x, device=self.device)
        else:
            theta_x = theta_x.to(self.device)
        is_batch = theta_x.dim() > 1 or (theta_x.dim() == 1 and theta_x.shape[0] > 1)
        batch_size = theta_x.shape[0] if is_batch else 1
        theta_x = theta_x.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H2 = torch.zeros((batch_size , size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        n = 1
        a = -1
        b = 0
        while n <= self.ny and a < size - 1 and b < size:
            a += 2
            b += 2
            if b < n * self.nx:
                H2[:, a, b] = -J_coe_tensor
                H2[:, b, a] = -J_coe_tensor.conj()
            else:
                n += 1
                if self.nx % 2 == 1 and n % 2 == 0:
                    a += 0
                    b += 0
                elif self.nx % 2 == 1 and n % 2 != 0:
                    a += 2
                    b += 2
                elif self.nx % 2 == 0:
                    a += 1
                    b += 1
                if a < size - 1 and b < size:
                    H2[:, a, b] = -J_coe_tensor
                    H2[:, b, a] = -J_coe_tensor.conj()
            if self.nx == 2:
                a += 1
                b += 1
            if b >= self.ny * self.nx - 2:
                break
        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            phase = torch.exp(1j * theta_x).squeeze()
            while self.nx - 1 + 2 * self.nx * p < size and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                H2[:, a, b] = -J_coe_tensor * phase
                H2[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        return H2.squeeze() if not is_batch else H2
    
    def Hamiltonian_tbc3(self, theta_y, tbc='y'):
        '''The time-independent Hamiltonian H3 for 2T/5 <= t < 3T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        if isinstance(theta_y, (int, float)):
            theta_y = torch.tensor([theta_y], device=self.device)
        elif not isinstance(theta_y, torch.Tensor):
            theta_y = torch.tensor(theta_y, device=self.device)
        else:
            theta_y = theta_y.to(self.device)
        is_batch = theta_y.dim() > 1 or (theta_y.dim() == 1 and theta_y.shape[0] > 1)
        batch_size = theta_y.shape[0] if is_batch else 1
        theta_y = theta_y.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H3 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        if self.nx % 2 == 1:  # odd nx
            for i in range(size):
                a = 2 * i + 1
                b = self.nx + 2 * i + 1
                if b < size:
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
        else:  # Even nx
            n = 1
            a = 1
            b = 1 + self.nx
            if b < size:
                H3[:, a, b] = -J_coe_tensor
                H3[:, b, a] = -J_coe_tensor.conj()
            while n < self.ny and a < size - 1 and b < size - 1:
                a += 2
                b += 2
                if a < n * self.nx:
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
                else:
                    n += 1
                    if n % 2 == 0:  # even n
                        a -= 1
                        b -= 1
                    elif n % 2 != 0 and b < size - 1:  # odd n
                        a += 1
                        b += 1
                    else:
                        a -= 2
                        b -= 2
                    H3[:, a, b] = -J_coe_tensor
                    H3[:, b, a] = -J_coe_tensor.conj()
        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 2 * p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                phase = torch.exp(1j * theta_y).squeeze()
                H3[:, int(a), int(b)] = -J_coe_tensor * phase
                H3[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        return H3.squeeze() if not is_batch else H3

    def Hamiltonian_tbc4(self, theta_x, tbc='x'):
        '''The time-independent Hamiltonian H4 for 3T/5 <= t < 4T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        if isinstance(theta_x, (int, float)):
            theta_x = torch.tensor([theta_x], device=self.device)
        elif not isinstance(theta_x, torch.Tensor):
            theta_x = torch.tensor(theta_x, device=self.device)
        else:
            theta_x = theta_x.to(self.device)
        is_batch = theta_x.dim() > 1 or (theta_x.dim() == 1 and theta_x.shape[0] > 1)
        batch_size = theta_x.shape[0] if is_batch else 1
        theta_x = theta_x.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H4 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        n = 1
        a = -2
        b = -1
        while n <= self.ny and a < size - 2 and b < size - 2:
            a += 2
            b += 2
            if b < n * self.nx:
                H4[:, a, b] = -J_coe_tensor
                H4[:, b, a] = -J_coe_tensor.conj()
            else:
                n += 1
                if self.nx % 2 == 0 and self.nx != 2:  # even nx
                    a += 1
                    b += 1
                elif self.nx % 2 == 0 and self.nx == 2 and b < size - 2:  # even nx and nx = 2
                    a += 2
                    b += 2
                    n += 1
                elif self.nx % 2 == 1 and n % 2 == 1:  # odd nx and n is odd
                    a += 0
                    b += 0
                elif self.nx % 2 == 1 and n % 2 == 0:  # odd nx and n is even
                    a += 2
                    b += 2
                else:
                    n += 1
                    a += -2
                    b += -2
                H4[:, a, b] = -J_coe_tensor
                H4[:, b, a] = -J_coe_tensor.conj()

        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while 2 * self.nx * (1 + p) - 1 < size and self.nx % 2 == 0:
                a = 2 * self.nx * (1 + p) - 1
                b = 2 * self.nx * p + self.nx
                phase = torch.exp(1j * theta_x).squeeze()
                H4[:, a, b] = -J_coe_tensor * phase
                H4[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        return H4.squeeze() if not is_batch else H4
    
    def aperiodic_Honsite(self, vdT, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, contourplot=False, save_path=None):
        '''Adding aperiodic potential to the onsite Hamiltonian'''
        '''The extra phi1_ex and phi2_ex is for the convenience of adding extra phase to the potential'''
        # Convert vdT to a tensor if it's not already one, using the recommended method
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)
        
        vd = vdT / self.T
        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)
        
        size = self.nx * self.ny
        H_aperiodic = torch.zeros((vd.shape[0], size, size), dtype=torch.cdouble, device=self.device)
        sites = self.lattice_numbering()

        # Ensure the rotation angle is on the right device
        rotation_angle = rotation_angle.to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)

        # Computing u and v, ensure they are tensors of appropriate dimensions
        x_indices = torch.linspace(-(self.nx-1)/2, (self.nx-1)/2, steps=self.nx, device=self.device).reshape(1, -1)
        y_indices = torch.linspace(-(self.ny-1)/2, (self.ny-1)/2, steps=self.ny, device=self.device).reshape(-1, 1)
        u = x_indices * torch.cos(rotation_angle) - y_indices * torch.sin(rotation_angle)
        v = x_indices * torch.sin(rotation_angle) + y_indices * torch.cos(rotation_angle)

        # Compute phase shifts using proper tensor operations
        phi1 = 2 * np.pi * (a * torch.cos(rotation_angle).item() - b * torch.sin(rotation_angle).item())
        phi2 = 2 * np.pi * (a * torch.sin(rotation_angle).item() + b * torch.cos(rotation_angle).item())

        # Calculate the potential and assign it correctly to the diagonal
        print('phi1 is', phi1_ex, 'phi2 is', phi2_ex)
        potential = torch.cos(2 * np.pi * u + phi1 + phi1_ex) + torch.cos(2 * np.pi * v + phi2 + phi2_ex)
        potential = potential.reshape(self.ny, self.nx).to(torch.cdouble)
        # print(potential)
        # Use broadcasting to apply the potential to all vd values
        H_aperiodic[:, sites.long(), sites.long()] = potential * -vd / 2

        H_ap = None  # Initialize H_ap to None or a default value
        if contourplot:
            # Use the first vdT for plotting
            H_ap = H_aperiodic[0].diag().cpu().numpy().reshape(self.ny, self.nx).real
            # print(H_ap)
            plt.figure(figsize=(8, 6))
            norm = plt.Normalize(-vd, vd)
            # print(np.min(H_ap), np.max(H_ap))
            cmap = plt.get_cmap('viridis')
            plt.imshow(H_ap, cmap=cmap, norm=norm, interpolation='nearest', origin='upper')

            fontsize = 24
            ticksize = 16

            cbar = plt.colorbar(aspect=50)
            cbar.set_label(r'$V_{\mathbf{r}}T$', fontsize=fontsize)

            # Get the current tick labels and locations
            tick_labels = cbar.ax.get_yticklabels()
            tick_locations = cbar.ax.get_yticks()

            # Prepare new tick labels
            new_tick_labels = []
            for tick in tick_labels:
                try:
                    # Attempt to convert tick text, handle possible formatting issues
                    label_value = float(tick.get_text().replace('−', '-').replace('−', '-'))
                    new_tick_labels.append(label_value * self.T)
                except ValueError:
                    # Handle possible conversion errors
                    new_tick_labels.append(0)  # Default to 0 or some suitable fallback value

            # Set the tick positions and labels on the colorbar
            cbar.ax.set_yticks(tick_locations)  # Ensure ticks are set with FixedLocator
            cbar.ax.set_yticklabels([f'{label:.3f}' for label in new_tick_labels])  # Set tick labels

            cbar.ax.tick_params(labelsize=ticksize)

            plt.xlabel('X', fontsize=fontsize)
            plt.ylabel('Y', fontsize=fontsize)

            # Change font size of x and y tick labels
            x_ticks = np.arange(0, self.nx, 4)
            y_ticks = np.arange(0, self.ny, 4)
            plt.xticks(x_ticks, fontsize=ticksize)
            plt.yticks(y_ticks, fontsize=ticksize)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
            return H_aperiodic.squeeze(), H_ap
        else:
            return H_aperiodic.squeeze()
    
    def compute_mu(self, args):
        x, y, a, b, phi1_ex, phi2_ex, theta = args
        u = x * torch.cos(theta) - y * torch.sin(theta)
        v = x * torch.sin(theta) + y * torch.cos(theta)
        phi1 = 2 * np.pi * (a * torch.cos(theta).item() - b * torch.sin(theta).item())
        phi2 = 2 * np.pi * (a * torch.sin(theta).item() + b * torch.cos(theta).item())
        return torch.cos(2 * np.pi * u + phi1 + phi1_ex) + torch.cos(2 * np.pi * v + phi2 + phi2_ex)
    
    def quasip_continuum(self, x, y, a, b, phi1_ex, phi2_ex, rotation_angle):
        lx = len(x)
        ly = len(y)
        mu = np.zeros((lx, ly), dtype=float)
        
        # Create a list of arguments for each combination of x[i] and y[j]
        args_list = [(x[i], y[j], a, b, phi1_ex, phi2_ex, rotation_angle) for i in range(lx) for j in range(ly)]
        
        # Use ThreadPoolExecutor to parallelize the computation, limiting the number of threads
        max_threads = min(32, len(args_list))  # Adjust the number of threads as needed
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = list(executor.map(self.compute_mu, args_list))
        # Reshape the results back into the mu matrix
        for index, value in enumerate(results):
            i = index % lx
            j = index // lx
            mu[i, j] = value
        return mu
        
    def visualise_quasiperiodic(self, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, grid_density=100, circle_size=0.1, circle_color='black', colorbar=False, save_path=None):
        """
        Visualize a 2D square lattice with nx x ny sites. Each lattice site is represented by a hollow circle with a dashed outline.
        The contour plot is centered at ((self.nx - 1)/2, (self.ny - 1)/2).
        """
        rotation_angle = rotation_angle.to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a denser grid for the contour plot, centered at ((self.nx - 1)/2, (self.ny - 1)/2)
        x_dense = np.linspace(-1, self.nx, grid_density) - (self.nx - 1) / 2
        y_dense = np.linspace(-1, self.ny, grid_density) - (self.ny - 1) / 2

        # Compute the quasiperiodic potential on the denser grid
        mu = self.quasip_continuum(x_dense, y_dense, a, b, phi1_ex, phi2_ex, rotation_angle)

        # Plot the contour
        X_contour, Y_contour = np.meshgrid(x_dense + (self.nx - 1) / 2, y_dense + (self.ny - 1) / 2, indexing='ij')
        contour_plot = ax.contourf(X_contour, Y_contour, mu, levels=15, cmap='viridis')

        # Create a colorbar with the same height as the main plot
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # Increased pad value to move the color bar further
            cbar = fig.colorbar(contour_plot, cax=cax)
            cbar.ax.tick_params(labelsize=20)  # Set font size for colorbar tick labels
            cbar.set_label(r'$V_{r}T$', fontsize=25)  # Set the label for the color bar

        # Set limits and aspect ratio
        ax.set_xlim(-1, self.nx)
        ax.set_ylim(-1, self.ny)
        ax.set_aspect('equal')

        # Set tick positions and labels for x and y axes, excluding -1
        x_ticks = np.arange(0, self.nx, 1)
        y_ticks = np.arange(0, self.ny, 1)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=25)  # Set font size for x-axis tick labels
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=25)  # Set font size for y-axis tick labels

        # Set x and y axis labels
        ax.set_xlabel('X', fontsize=30)
        ax.set_ylabel('Y', fontsize=30)

        # Generate coordinates for the lattice points using torch
        x_coords, y_coords = torch.meshgrid(torch.arange(self.nx), torch.arange(self.ny), indexing='ij')

        # Flatten the coordinates for easier plotting
        x_coords = x_coords.flatten().cpu().numpy()
        y_coords = y_coords.flatten().cpu().numpy()

        # Plot hollow circles at each lattice point with specified color
        for x, y in zip(x_coords, y_coords):
            circle = plt.Circle((x, y), circle_size, edgecolor=circle_color, facecolor='none', linestyle='--', linewidth=1.5)
            ax.add_patch(circle)

        # Remove axis labels and ticks if needed (commented out here)
        # ax.set_xticks([])
        # ax.set_yticks([])

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
    
    def check_diagonal_symmetry(self, H_ap):
        """
        Check if the matrix H_ap is symmetric along the main diagonal and the anti-diagonal.
        
        Args:
        - H_ap (numpy.ndarray): A 2D numpy array representing the potential on a lattice.
        
        Returns:
        - tuple: (main_diagonal_symmetry, anti_diagonal_symmetry)
        - main_diagonal_symmetry (bool): True if H_ap is symmetric about the main diagonal.
        - anti_diagonal_symmetry (bool): True if H_ap is symmetric about the anti-diagonal.
        """
        # Symmetry across the main diagonal (top-left to bottom-right)
        main_diagonal_symmetry = np.allclose(H_ap, H_ap.T)
        
        # Symmetry across the anti-diagonal (top-right to bottom-left)
        # Flip the matrix along the vertical axis and then check for main diagonal symmetry
        flipped_H_ap = np.fliplr(H_ap)
        anti_diagonal_symmetry = np.allclose(flipped_H_ap, flipped_H_ap.T)
        
        return main_diagonal_symmetry, anti_diagonal_symmetry
    
    def Hamiltonian_disorder(self, vdT, contourplot=False, initialise=False, save_path=None):
        '''The disorder Hamiltonian adding random onsite potential to the total Hamiltonian for which is uniformly distributed in the range (-vd, vd)'''
        # Convert vd to a tensor if it's not already one, using the recommended method
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)
        
        vd = vdT / self.T
        # print(vd)
        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)
        
        if self.H_disorder_cached is None or initialise:
            size = self.nx * self.ny
            random_values = torch.rand(size, device=self.device) * 2 - 1  # Uniform distribution between -1 and 1
            # print('random_values', random_values)
            self.H_disorder_cached = torch.diag(random_values)

        # Use broadcasting to apply vd to the cached disorder matrix
        disorder_matrix = self.H_disorder_cached * vd

        if contourplot:
            # Use the first vd for plotting
            H_dis = disorder_matrix[0].diag().cpu().numpy().reshape(self.ny, self.nx)
            
            plt.figure(figsize=(8, 6))
            norm = plt.Normalize(-vd, vd)
            cmap = plt.get_cmap('viridis')
            plt.imshow(H_dis, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
            fontsize = 24
            ticksize = 16
            cbar = plt.colorbar(aspect=50)
            cbar.set_label(r'$V_{\mathbf{r}}T$', fontsize=fontsize)
            
            # Get the current tick labels and locations
            tick_labels = cbar.ax.get_yticklabels()
            tick_locations = cbar.ax.get_yticks()

            # Multiply the tick labels by self.T
            new_tick_labels = [float(tick.get_text().replace('−', '-')) * self.T for tick in tick_labels]
            # Set the new tick labels on the colorbar
            cbar.ax.set_yticks(tick_locations)  # Set tick locations
            cbar.ax.set_yticklabels([f'{label:.3f}' for label in new_tick_labels])  # Set tick labels
            cbar.ax.tick_params(labelsize=ticksize)
            plt.xlabel('X', fontsize=fontsize)
            plt.ylabel('Y', fontsize=fontsize)
            
            # Change font size of x and y tick labels
            x_ticks = np.arange(0, self.nx, 1)
            y_ticks = np.arange(0, self.ny, 1)
            plt.xticks(x_ticks, fontsize=ticksize)
            plt.yticks(y_ticks, fontsize=ticksize)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
        return disorder_matrix.squeeze()
    
    def Hamiltonian_onsite(self, vdT, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True, contourplot=False):
        '''The time-independent Hamiltonian H5 for 4T/5 <= t < T with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        size = self.nx * self.ny
        # Convert vd to a tensor if it's not already one
        if not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)
        
        # Reshape vd for broadcasting
        vdT = vdT.reshape(-1, 1, 1)

        # Create H_onsite with an additional dimension for batch processing
        H_onsite = torch.zeros((vdT.shape[0], size, size), dtype=torch.cdouble, device=self.device)

        if delta is None:
            delta = self.delta_AB

        deltas = torch.full((size,), delta, dtype=torch.cdouble, device=self.device)
        deltas[1::2] *= -1  # Alternate the sign starting from the second element

        # Adjust the sign based on row blocks
        for n in range(self.ny):
            if n % 2 == 1:
                start_idx = n * self.nx
                end_idx = start_idx + self.nx
                deltas[start_idx:end_idx] *= -1

        # Add deltas to the diagonal for all batches
        H_onsite[:, torch.arange(size), torch.arange(size)] = deltas

        if fully_disorder:
            H_dis = self.Hamiltonian_disorder(vdT, contourplot=contourplot, initialise=initialise)
        else:
            rotation_angle_tensor = rotation_angle.clone().detach().to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)
            H_dis = self.aperiodic_Honsite(vdT, rotation_angle_tensor, a, b, phi1_ex, phi2_ex, contourplot=contourplot)
        H5 = H_onsite + H_dis
        return H5.squeeze()
    
    def Hamiltonian_tbc(self, t, tbc, vdT, rotation_angle, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        """The Hamiltonian H(t) with twisted boundary conditions in either x, y, or both x and y directions in the real space """
        
        # Ensure t is within [0, T)
        t = t % self.T

        # Convert inputs to tensors and move them to the correct device
        vdT = torch.as_tensor(vdT, device=self.device).reshape(-1, 1, 1, 1, 1)
        theta_x = torch.as_tensor(theta_x, device=self.device).reshape(1, -1, 1, 1, 1)
        theta_y = torch.as_tensor(theta_y, device=self.device).reshape(1, 1, -1, 1, 1)
        rotation_angle = torch.as_tensor(rotation_angle, device=self.device)

        size = self.nx * self.ny

        # Compute H_onsite
        H_onsite = self.Hamiltonian_onsite(vdT.squeeze(), rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        H_onsite = H_onsite.view(vdT.shape[0], 1, 1, size, size)

        # Compute H_tbc based on the time t
        if t < self.T/5:
            H_tbc = self.Hamiltonian_tbc1(theta_y.squeeze(), tbc)
            H_tbc = H_tbc.view(1, 1, -1, size, size)
        elif self.T/5 <= t < 2*self.T/5:
            H_tbc = self.Hamiltonian_tbc2(theta_x.squeeze(), tbc)
            H_tbc = H_tbc.view(1, -1, 1, size, size)
        elif 2*self.T/5 <= t < 3*self.T/5:
            H_tbc = self.Hamiltonian_tbc3(theta_y.squeeze(), tbc)
            H_tbc = H_tbc.view(1, 1, -1, size, size)
        elif 3*self.T/5 <= t < 4*self.T/5:
            H_tbc = self.Hamiltonian_tbc4(theta_x.squeeze(), tbc)
            H_tbc = H_tbc.view(1, -1, 1, size, size)
        else:  # 4*self.T/5 <= t < self.T
            H_tbc = torch.zeros(1, 1, 1, size, size, dtype=torch.cdouble, device=self.device)

        # Add H_tbc to H_onsite and broadcast to final shape
        H = (H_onsite + H_tbc).expand(vdT.shape[0], theta_x.shape[1], theta_y.shape[2], size, size)

        return H.squeeze()
    
    def time_evolution_operator(self, t, tbc, vdT, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        '''The time evolution operator U(t) = exp(-iH(t))
        n is the order of expansion of the time evolution operator'''
        
        # Convert vd to a tensor if it's not already one
        if not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)

        # Reshape vd for broadcasting
        vdT = vdT.reshape(-1, 1, 1)

        H_onsite = self.Hamiltonian_onsite(vdT, rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_y, tbc).unsqueeze(0) + H_onsite
        H2 = self.Hamiltonian_tbc2(theta_x, tbc).unsqueeze(0) + H_onsite
        H3 = self.Hamiltonian_tbc3(theta_y, tbc).unsqueeze(0) + H_onsite
        H4 = self.Hamiltonian_tbc4(theta_x, tbc).unsqueeze(0) + H_onsite
        H5 = H_onsite

        if t < self.T/5:
            U = torch.matrix_exp(-1j * t * H1)
        elif self.T/5 <= t < 2 * self.T/5:
            U1 = torch.matrix_exp(-1j * (self.T/5) * H1)
            U2 = torch.matrix_exp(-1j * (t - self.T/5) * H2)
            U = U2 @ U1
        elif 2 * self.T/5 <= t < 3 * self.T/5:
            U1 = torch.matrix_exp(-1j * (self.T/5) * H1)
            U2 = torch.matrix_exp(-1j * (self.T/5) * H2)
            U3 = torch.matrix_exp(-1j * (t - 2 * self.T/5) * H3)
            U = U3 @ U2 @ U1
        elif 3 * self.T/5 <= t < 4 * self.T/5:
            U1 = torch.matrix_exp(-1j * (self.T/5) * H1)
            U2 = torch.matrix_exp(-1j * (self.T/5) * H2)
            U3 = torch.matrix_exp(-1j * (self.T/5) * H3)
            U4 = torch.matrix_exp(-1j * (t - 3 * self.T/5) * H4)
            U = U4 @ U3 @ U2 @ U1
        elif 4 * self.T/5 <= t <= self.T:
            U1 = torch.matrix_exp(-1j * (self.T/5) * H1)
            U2 = torch.matrix_exp(-1j * (self.T/5) * H2)
            U3 = torch.matrix_exp(-1j * (self.T/5) * H3)
            U4 = torch.matrix_exp(-1j * (self.T/5) * H4)
            U5 = torch.matrix_exp(-1j * (t - 4 * self.T/5) * H5)
            U = U5 @ U4 @ U3 @ U2 @ U1

        return U
    
    def time_evolution_operator1(self, t, steps_per_segment, tbc, vdT, rotation_angle, theta_x, theta_y, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):
        '''Time evolution operator for time t ≤ T with a specified number of steps per T/5 segment'''
        '''Support not only scalar (VdT, theta_x, theta_y, t) but also batch processing for multiple (vdT, theta_x, theta_y, t): vectorization of VdT, theta_x, theta_y, and t: 1D tensors
        the output shape is then (N_VdT, N_thetax, N_thetay, N_t, nx*ny, nx*ny)'''
        
        # Convert inputs to tensors and move them to the correct device
        vdT = torch.as_tensor(vdT, device=self.device).reshape(-1, 1, 1, 1, 1, 1)
        theta_x = torch.as_tensor(theta_x, device=self.device).reshape(1, -1, 1, 1, 1, 1)
        theta_y = torch.as_tensor(theta_y, device=self.device).reshape(1, 1, -1, 1, 1, 1)
        t = torch.as_tensor(t, device=self.device).reshape(-1, 1)

        N_VdT, N_thetax, N_thetay, N_t = vdT.shape[0], theta_x.shape[1], theta_y.shape[2], t.shape[0]
        # print("N_VdT: ", N_VdT)
        # print("N_thetax: ", N_thetax)
        # print("N_thetay: ", N_thetay)
        # print("N_t: ", N_t)
        size = self.nx * self.ny

        # Calculate dt based on the number of steps per segment
        # print(steps_per_segment)
        dt = self.T / (5 * steps_per_segment)
        # print('dt',dt)
        # Compute H_onsite for all Vd values
        H_onsite = self.Hamiltonian_onsite(vdT.squeeze(), rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        H_onsite = H_onsite.view(N_VdT, 1, 1, 1, size, size).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        # print("H_onsite shape: ", H_onsite.shape)
        # Compute H1, H2, H3, H4 for all theta_x and theta_y values
        H1 = self.Hamiltonian_tbc1(theta_y.squeeze(), tbc).view(1, 1, N_thetay, 1, size, size).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        H2 = self.Hamiltonian_tbc2(theta_x.squeeze(), tbc).view(1, N_thetax, 1, 1, size, size).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        H3 = self.Hamiltonian_tbc3(theta_y.squeeze(), tbc).view(1, 1, N_thetay, 1, size, size).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        H4 = self.Hamiltonian_tbc4(theta_x.squeeze(), tbc).view(1, N_thetax, 1, 1, size, size).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        # print("H4 shape: ", H4.shape)
        # Move all tensors to the specified device
        H_onsite = H_onsite.to(self.device)
        H1 = H1.to(self.device)
        H2 = H2.to(self.device)
        H3 = H3.to(self.device)
        H4 = H4.to(self.device)
        identity = torch.eye(size, dtype=torch.cdouble, device=self.device).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)
        U = torch.eye(size, dtype=torch.cdouble, device=self.device).expand(N_VdT, N_thetax, N_thetay, N_t, size, size)

        total_steps = torch.floor(t / dt).long()
        max_steps = total_steps.max().item()
        # print('t',t)
        # print('dt', dt)
        # print(total_steps)
        for step in range(max_steps):
            current_t = step * dt
            current_t_mod = current_t % self.T
            active = step < total_steps
            # print("Shape of 'active' before view:", active.shape)
            # print("Total number of elements in 'active':", active.numel())
            active = active.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
            mask1 = (current_t_mod < self.T/5) & active
            mask2 = (current_t_mod >= self.T/5) & (current_t_mod < 2*self.T/5) & active
            mask3 = (current_t_mod >= 2*self.T/5) & (current_t_mod < 3*self.T/5) & active
            mask4 = (current_t_mod >= 3*self.T/5) & (current_t_mod < 4*self.T/5) & active
            mask5 = (current_t_mod >= 4*self.T/5) & active
            H0 = torch.zeros_like(H1).to(self.device)
            Hr = torch.zeros_like(H1).to(self.device)
            Hr[mask1] = H1[mask1]
            Hr[mask2] = H2[mask2]
            Hr[mask3] = H3[mask3]
            Hr[mask4] = H4[mask4]
            Hr[mask5] = H0[mask5]
            combined_mask = mask1 | mask2 | mask3 | mask4 | mask5
            combined_mask = combined_mask.to(self.device)
            combined_mask = combined_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, size, size)
            U_step = torch.where(combined_mask, 
                                self.infinitesimal_evol_operator(Hr, H_onsite, dt),
                                identity)
            U = U_step @ U
        # Handle any remaining time
        remaining_time = t - total_steps * dt
        mask1 = (t < self.T/5) & (remaining_time > 0)
        mask2 = (t >= self.T/5) & (t < 2*self.T/5) & (remaining_time > 0)
        mask3 = (t >= 2*self.T/5) & (t < 3*self.T/5) & (remaining_time > 0)
        mask4 = (t >= 3*self.T/5) & (t < 4*self.T/5) & (remaining_time > 0)
        mask5 = (t >= 4*self.T/5) & (remaining_time > 0)
        mask1 = mask1.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
        mask2 = mask2.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
        mask3 = mask3.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
        mask4 = mask4.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
        mask5 = mask5.view(1, 1, 1, N_t).expand(N_VdT, N_thetax, N_thetay, -1)
        
        H0 = torch.zeros_like(H1)
        Hr = torch.zeros_like(H1)
        
        Hr[mask1] = H1[mask1]
        Hr[mask2] = H2[mask2]
        Hr[mask3] = H3[mask3]
        Hr[mask4] = H4[mask4]
        Hr[mask5] = H0[mask5]
        
        combined_mask = mask1 | mask2 | mask3 | mask4 | mask5
        combined_mask = combined_mask.to(self.device)
        combined_mask = combined_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, size, size)
        remaining_time = remaining_time.view(1, 1, 1, N_t, 1, 1).expand(N_VdT, N_thetax, N_thetay, -1, size, size)
        U_step = torch.where(combined_mask, 
                            self.infinitesimal_evol_operator(Hr, H_onsite, remaining_time),
                            identity)
        U = U_step @ U
        # U.bmm_(U_step)
        # print("U shape: ", U.shape)
        # After the loop, if H1, H2, H3, H4 are no longer needed:
        torch.cuda.synchronize()
        del H1, H2, H3, H4
        torch.cuda.empty_cache()  # If using GPU

        # After using H_onsite for the last time:
        torch.cuda.synchronize()

        del H_onsite
        torch.cuda.empty_cache()  # If using GPU
        return U.squeeze()
    
    ## Exploring the bulk properties of the system
    ## GOAL 1. Quasienergies and wavefuntion -- COMPLETED
    ## GOAL 2: Effective Hamiltonian_bulk -- COMPLETED
    ## GOAL 3: Deformed Time evolution operator U_epsilon -- COMPLETED
    ## GOAL 4. The Winding number of the quasienergy gaps and the Chern number of the bulk bands -- COMPLETED
    ## GOAL 5. Level spacing statistics of the bulk evolution operator --COMPLETED
    ## GOAL 6. The Inverse Participation Ratios
    
    def eigen_grid(self, vdT_tensor, N_div, steps_per_segment, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''Version 2
        The eigenvalues and eigenvectors on the grid of the quasienergy spectrum'''
        # Convert vd to a tensor if it's not already one, using the recommended method
        theta_x = torch.linspace(0, 2*torch.pi, N_div+1, device=self.device)
        theta_y = torch.linspace(0, 2*torch.pi, N_div+1, device=self.device)
        t = torch.linspace(0, self.T, N_div+1, device=self.device)
        U_tensor = self.time_evolution_operator1(t, steps_per_segment, 'xy', vdT_tensor, rotation_angle, theta_x, theta_y, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        eigvals, eigvecs = torch.linalg.eig(U_tensor)
        # print("Original eigvecs shape:", eigvecs.shape)
        # print("Original eigvecs norm:", torch.norm(eigvecs, dim=-2))
        # print("Original eigvecs (first point):", eigvecs[-5, -5, -5])
        # Free up memory
        del U_tensor
        torch.cuda.empty_cache()
        # Compute the real part of the eigenvalues for sorting
        eigv = -1j * torch.log(eigvals)
        # print(eigv)
        eigv_r = eigv.real
        
        # Free up memory
        del eigv
        torch.cuda.empty_cache()

        # Sort the eigenvalues based on their real parts
        sorted_indices = torch.argsort(eigv_r, dim=-1)
        
        # Reorder the eigenvalues, their real parts of the log(eigenvalues), and the eigenvectors
        sorted_eigvals = torch.gather(eigvals, -1, sorted_indices)
        sorted_eigv_r = torch.gather(eigv_r, -1, sorted_indices)
        expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
        sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)
        # print("First sorted eigenvector (for verification):", sorted_eigvecs[-5, -5, -5])
        # print("Sorted eigenvectors shape:", sorted_eigvecs.shape)
        # # Check orthogonality
        # dot_products = torch.matmul(sorted_eigvecs.transpose(-1, -2).conj(), sorted_eigvecs)
        # off_diagonal = dot_products - torch.eye(dot_products.shape[-1], device=dot_products.device)
        # print("Max off-diagonal element:", torch.max(torch.abs(off_diagonal)))
        # print("Sorted eigenvectors norm:", torch.norm(sorted_eigvecs, dim=-2))
        # Free up memory
        del eigvals, eigv_r, eigvecs, sorted_indices, expanded_indices
        torch.cuda.empty_cache()
        if plot == True:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
            def update(frame):
                ax.clear()
                ax.set_ylim(0, 1.1)
                
                # Get eigenvalues for this time frame
                eigvals_t = sorted_eigvals[:, :, frame, :].reshape(-1)
                
                # Debugging prints
                # print(f"Frame {frame}: Min abs: {torch.min(torch.abs(eigvals_t))}, Max abs: {torch.max(torch.abs(eigvals_t))}")
                # print(f"Number of eigenvalues: {eigvals_t.numel()}")
                
                # Convert to numpy and handle potential complex numbers
                eigvals_np = eigvals_t.cpu().numpy()
                angles = np.angle(eigvals_np)
                magnitudes = np.abs(eigvals_np)
                
                # Plot eigenvalues on complex unit circle
                scatter = ax.scatter(angles, magnitudes, alpha=0.5, s=30)
                
                ax.set_title(f'Eigenvalues at t = {t[frame].item():.2f}')
                return scatter,
            
            ani = animation.FuncAnimation(fig, update, frames=N_div+1, blit=True)
            
            if save_path is not None:
                ani.save(save_path, writer='ffmpeg')
            plt.show()
        return sorted_eigv_r, sorted_eigvals, sorted_eigvecs

    def animate_time_spectra(self, vdT_value, N_div, steps_per_segment, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, fps=5, filename= None):
        ## Here, Vd should be a fixed value and t should be varied
        sorted_eigv_r, sorted_eigvals, _ = self.eigen_grid(vdT_value, N_div, steps_per_segment, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        theta_x = torch.linspace(0, 2*torch.pi, N_div+1, device=self.device).cpu().numpy()
        theta_y = torch.linspace(0, 2*torch.pi, N_div+1, device=self.device).cpu().numpy()
        t = torch.linspace(0, self.T, N_div+1, device=self.device)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='polar')
        def update(frame):
            ax1.clear()
            ax2.clear()
            # 3D quasienergy plot
            kx, ky = np.meshgrid(theta_x, theta_y)
            z = sorted_eigv_r[:, :, frame].cpu().numpy()
            for i in range(z.shape[-1]):
                ax1.plot_surface(kx, ky, z[:, :, i], cmap='viridis')
            ax1.set_xlabel(r'$\theta_{x}$')
            ax1.set_ylabel(r'$\theta_{y}$')
            ax1.set_zlabel(r'$T\epsilon$')
            ax1.set_title(f'Time Shot: {frame}')
            ax1.set_zlim(-torch.pi, torch.pi)
            ax1.view_init(elev=2, azim=5)
            # Polar plot of eigenvalues
            eigvals_t = sorted_eigvals[:, :, frame, :].reshape(-1)
            # Convert to numpy and handle potential complex numbers
            eigvals_np = eigvals_t.cpu().numpy()
            angles = np.angle(eigvals_np)
            magnitudes = np.abs(eigvals_np)
            # Plot eigenvalues on complex unit circle
            scatter = ax2.scatter(angles, magnitudes, alpha=0.5, s=30)
            ax2.set_ylim(0, 1.1)
            ax2.set_yticks([1])  # Only show tick at r=1
            ax2.set_title(f'Eigenvalues at t = {t[frame].item():.2f}')
            
            return ax1, scatter
        
        ani = animation.FuncAnimation(fig, update, frames=sorted_eigv_r.shape[2], interval=1000/fps, blit=False)
        
        if filename is not None:
            ani.save(filename, writer='ffmpeg')
        plt.show()
    
    def animate_aperiodic_spectra(self, vdT, N_div, steps_per_segment, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, fps=5, filename= None):
        '''Here, Vd should be tensor and t should be fixed'''
        thetax, thetay = self.get_theta_values(N_div, 'xy')
        sorted_eigv_r, sorted_eigvals, _ = self.quasienergies_states_bulk(steps_per_segment, vdT, thetax, thetay, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        theta_x = thetax.cpu().numpy()
        theta_y = thetay.cpu().numpy()
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='polar')
        def update(frame):
            ax1.clear()
            ax2.clear()
            # 3D quasienergy plot
            kx, ky = np.meshgrid(theta_x, theta_y)
            z = sorted_eigv_r[frame, :, :].cpu().numpy()
            for i in range(z.shape[-1]):
                ax1.plot_surface(kx, ky, z[:, :, i], cmap='viridis')
            ax1.set_xlabel(r'$\theta_{x}$')
            ax1.set_ylabel(r'$\theta_{y}$')
            ax1.set_zlabel(r'$T\epsilon$')
            vd = vdT / self.T
            ax1.set_title(f'Aperiodic Strength: {vd[frame]} T')
            ax1.set_zlim(-torch.pi, torch.pi)
            ax1.view_init(elev=2, azim=5)
            # Polar plot of eigenvalues
            eigvals_t = sorted_eigvals[frame, :, :, :].reshape(-1)
            # Convert to numpy and handle potential complex numbers
            eigvals_np = eigvals_t.cpu().numpy()
            angles = np.angle(eigvals_np)
            magnitudes = np.abs(eigvals_np)
            # Plot eigenvalues on complex unit circle
            scatter = ax2.scatter(angles, magnitudes, alpha=0.5, s=30)
            ax2.set_ylim(0, 1.1)
            ax2.set_yticks([1])  # Only show tick at r=1
            ax2.set_title(f'Eigenvalues at frame = {frame}')
            return ax1, scatter
        
        ani = animation.FuncAnimation(fig, update, frames=sorted_eigv_r.shape[2], interval=1000/fps, blit=False)
        
        if filename is not None:
            ani.save(filename, writer='ffmpeg')
        plt.show()
    
    def U_nu(self, s_nu_pi, s_nu_pj):
        """Version 1: Support batch processing: Checked
        Compute U_nu as defined in the paper.

        Parameters:
        s_nu_pi (torch.Tensor): Eigenvector at point pi
        s_nu_pj (torch.Tensor): Eigenvector at point pj

        Returns:
        torch.Tensor: U_nu value
        """
        inner_product = torch.sum(s_nu_pi.conj() * s_nu_pj, dim=0)
        # print(inner_product)
        abs_inner_product = torch.abs(inner_product)
        # print(abs_inner_product)
        result = inner_product / abs_inner_product
        return result
    
    def mod(self, a, b):
        return ((a - 1) % b) + 1

    def face_F_hat(self, i, j, k, N_div, eigvals, eigvecs):
        """Version 2: Batch processing of three alpha values 
        Compute F̂νp,α for all three faces (α = 1, 2, 3) simultaneously.
    
        Parameters:
        i, j, k (int): Indices of the base point of the faces
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        N_div (int): Number of divisions in each dimension
        Returns:
        torch.Tensor: F̂νp,α values for all three faces, shape (3, num_bands)
        """
        # Define the vertices of the faces for all three alphas
        ## Version 1
        # vertices = torch.tensor([
        #     # alpha = 1
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, (k+1)%N_div], 
        #     [i%N_div, j%N_div, (k+1)%N_div]],
        #     # alpha = 2
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [i%N_div, j%N_div, (k+1)%N_div], 
        #     [(i+1)%N_div, j%N_div, (k+1)%N_div], 
        #     [(i+1)%N_div, j%N_div, k%N_div]],
        #     # alpha = 3
        #     [[i%N_div, j%N_div, k%N_div], 
        #     [(i+1)%N_div, j%N_div, k%N_div], 
        #     [(i+1)%N_div, (j+1)%N_div, k%N_div], 
        #     [i%N_div, (j+1)%N_div, k%N_div]]
        # ], dtype=torch.long, device=self.device)
        ## Version 2
        vertices = torch.tensor([
            # alpha = 1
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k+1, N_div)], 
            [self.mod(i, N_div), self.mod(j, N_div), self.mod(k+1, N_div)]],
            # alpha = 2
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j, N_div), self.mod(k+1, N_div)], 
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k+1, N_div)], 
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k, N_div)]],
            # alpha = 3
            [[self.mod(i, N_div), self.mod(j, N_div), self.mod(k, N_div)],
            [self.mod(i+1, N_div), self.mod(j, N_div), self.mod(k, N_div)], 
            [self.mod(i+1, N_div), self.mod(j+1, N_div), self.mod(k, N_div)], 
            [self.mod(i, N_div), self.mod(j+1, N_div), self.mod(k, N_div)]]
        ], dtype=torch.long, device=self.device)
        
        # print("vertices for face", vertices)
        # Get eigenvectors at each vertex for all faces
        v = [[eigvecs[tuple(v)] for v in face] for face in vertices]
        # Compute U_nu for each edge of each face
        U12 = torch.stack([self.U_nu(v[alpha][0], v[alpha][1]) for alpha in range(3)])
        U23 = torch.stack([self.U_nu(v[alpha][1], v[alpha][2]) for alpha in range(3)])
        U34 = torch.stack([self.U_nu(v[alpha][2], v[alpha][3]) for alpha in range(3)])
        U41 = torch.stack([self.U_nu(v[alpha][3], v[alpha][0]) for alpha in range(3)])
        
        # Compute F̂νp,α
        F_hat = (1 / (2 * torch.pi * 1j)) * torch.log(U12 * U23 * U34 * U41)
        return F_hat.real
    
    def cube(self, i, j, k, N_div, eigvals, eigvecs):
        """
        Compute the cube function as defined in equation 4.3 of the paper for all bands simultaneously.
        
        Parameters:
        i, j, k (int): Indices of the base point of the cube
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        N_div (int): Number of divisions in each dimension
        
        Returns:
        torch.Tensor: Cube function values for each band
        """
        F_p = self.face_F_hat(i, j, k, N_div, eigvals, eigvecs)
        
        C_p = torch.zeros(F_p.shape[1], dtype=torch.float64, device=self.device)
        
        for alpha in range(3):
            i_plus, j_plus, k_plus = i, j, k
            if alpha == 0:
                i_plus = (i + 1)
            elif alpha == 1:
                j_plus = (j + 1)
            else:  # alpha == 2
                k_plus = (k + 1)
            
            F_p_plus = self.face_F_hat(i_plus, j_plus, k_plus, N_div, eigvals, eigvecs)
            
            C_p += F_p_plus[alpha] - F_p[alpha] # Modified (Deviated) from the algorithm in the paper
        
        C_p = torch.round(C_p)
        
        # if torch.sum(C_p) != 0:
        #     print(f"Warning: The sum of the elements in C_p is non-zero at cube ({i}, {j}, {k}).")
        
        return C_p
    
    def determine_m(self, i, j, k, eigvals):
        """Version 2
        Determine m^nu_p,alpha for all alpha and all bands simultaneously
        
        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        
        Returns:
        torch.Tensor: m^nu_p,alpha for all alphas and all bands, shape (3, num_bands)
        """
        # Get eigenvalues at p
        phi_p = eigvals[i, j, k]
        
        # Calculate indices for p + δ_alpha for all three directions
        indices_plus = torch.tensor([
            [(i + 1), j, k],
            [i, (j + 1), k],
            [i, j, (k + 1)]], dtype=torch.long, device=self.device)
            
        # Get eigenvalues at p - δ_alpha for all directions
        phi_p_plus_delta = eigvals[indices_plus[:, 0], indices_plus[:, 1], indices_plus[:, 2]]
        # Calculate the difference for all directions
        diff = (phi_p - phi_p_plus_delta)
        
        # Determine m for all alphas and all bands
        m = - torch.floor((diff + torch.pi) / (2 * torch.pi))
        # print(m)
        return m.long()
        
    def determine_M(self, i, j, k, eigvals, C_p):
        """Version 2
        Determine M^ν_p when Ĉ^ν_p ≠ 0 for two indices ν, ν'
        
        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        C_p (torch.Tensor): Ĉ^ν_p values
        
        Returns:
        torch.Tensor: M^ν_p values for all bands
        """
        # Get eigenvalues at p
        phi_p = eigvals[i, j, k]
        M_p = torch.zeros_like(C_p)
        non_zero_indices = torch.nonzero(C_p).squeeze()
        
        if len(C_p) == 2 and len(non_zero_indices) == 2:
            nu, nu_prime = non_zero_indices
            diff = phi_p[nu] - phi_p[nu_prime]
            M_nu = - torch.floor((diff + torch.pi) / (2 * torch.pi))
            M_p[nu] = M_nu
        else:
            # print(C_p)
            for i in range(len(C_p)):
                j = (i+1) % len(C_p)
                if C_p[i] == - C_p[j] and C_p[i] != 0:
                    diff = phi_p[i] - phi_p[j]
                    M_nu = - torch.floor((diff + torch.pi) / (2 * torch.pi))
                    M_p[i] = M_nu
        return M_p
        
    def w3(self, vdT_value, N_div, steps_per_segment, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True):
        eigvals, eigenvalss, eigvecs = self.eigen_grid(vdT_value, N_div, steps_per_segment, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        n_band = self.nx * self.ny
        w3 = 0
        delta = 1/N_div
        delta_space = delta * torch.pi * 2
        delta_t = delta * self.T
        for i in range(N_div):
            for j in range(N_div): 
                for k in range(N_div): 
                    # print(r"($i_1, i_2, i_3$)", i,j,k)
                    p = torch.tensor([delta_space * (i), delta_space * (j), delta_t * (k)], dtype=torch.float64, device=self.device)
                    # print('p', p)
                    C_p = self.cube(i, j, k, N_div, eigvals, eigvecs)
                    # print(r'$C_p$', C_p)
                    M_p = self.determine_M(i, j, k, eigvals, C_p)
                    # print(r'$M_p$', M_p)
                    F_p = self.face_F_hat(i, j, k, N_div, eigvals, eigvecs)
                    m_p = self.determine_m(i, j, k, eigvals)
                    if torch.any(M_p != 0):
                        print(f"($i_1, i_2, i_3$)", i,j,k)
                        print('p', p)
                        print(f'$C_p$', C_p)
                        print(f'$M_p$', M_p)
                        print(f'$F_p$', F_p)
                        print(f'$m_p$', m_p)
                        print("\n")
                    # if torch.any(m_p != 0):
                    #     print(f'$F_p$', F_p)
                    #     print(f'$m_p$', m_p)
                    #     print("\n")
                    # Update W3
                    w3 += torch.sum(C_p * M_p) + torch.sum(F_p * m_p)
        return w3, eigenvalss, eigvecs
    
    def log_with_branch_cut(self, z, branch_cut_angle=0):
        # Ensure the branch cut angle is a tensor
        if not isinstance(branch_cut_angle, torch.Tensor):
            branch_cut_angle = torch.tensor([branch_cut_angle], dtype=torch.float64, device=z.device)
        else:
            # Ensure branch_cut_angle is on the same device as z
            branch_cut_angle = branch_cut_angle.to(device=z.device, dtype=torch.float64)
        # Step 1: Compute the magnitude of z
        magnitude = torch.abs(z)
        # Step 2: Compute the initial phase
        initial_phase = torch.angle(z)
        
        # Normalize the branch_cut_angle to be within [-pi, pi)
        branch_cut_angle = (branch_cut_angle + torch.pi) % (2 * torch.pi) - torch.pi

        # Step 3: Adjust phase to be within [branch_cut_angle, branch_cut_angle + 2*pi)
        adjusted_phase = initial_phase.unsqueeze(-1) - branch_cut_angle
        adjusted_phase = (adjusted_phase + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi)
        adjusted_phase += branch_cut_angle  # Shift back to the desired range

        # Correct phases that are not within the specified range
        adjusted_phase += torch.where(adjusted_phase < branch_cut_angle, 2 * torch.pi, 0)
        adjusted_phase -= torch.where(adjusted_phase >= branch_cut_angle + 2 * torch.pi, 2 * torch.pi, 0)

        # Step 4: Compute the logarithm using the new angle and magnitude
        log_z = torch.log(magnitude).unsqueeze(-1) + 1j * adjusted_phase
        
        return log_z
    
    def determine_K(self, i, j, k, eigvals, eigvecs, branch_cut_angle):
        """
        Determine K^ν_(i,j) such that |-i log_ξ d^ν(p) - φ^ν_p + 2πK^ν_(i,j)| < π at i3 = N for all bands simultaneously

        Parameters:
        i, j, k (int): Indices of the current point in parameter space
        eigvals (torch.Tensor): Pre-computed eigenvalues from eigen_grid
        eigvecs (torch.Tensor): Pre-computed eigenvectors from eigen_grid
        branch_cut_angle (float): Angle for the branch cut in radians

        Returns:
        torch.Tensor: K^ν_(i,j) values for all bands
        """
        # Get eigenvalues at p
        d_p = eigvals[i, j, k]
        # Compute φ^ν_p using -i log(d^nu_p)
        phi_p = -1j * torch.log(d_p)
        # print("without branch cut",phi_p)
        # Compute -i log_ξ d^ν(p) using the provided branch cut angle
        log_xi_d = -1j * self.log_with_branch_cut(d_p, branch_cut_angle)
        # print('with branch cut', log_xi_d)
        # Compute the difference
        # Handle both scalar and tensor branch_cut_angle
        if log_xi_d.dim() > phi_p.dim():
            diff = log_xi_d - phi_p.unsqueeze(-1)
        else:
            diff = log_xi_d - phi_p
        # print("diff", diff)
        # Determine K
        K = -torch.floor((diff.real + torch.pi) / (2 * torch.pi))
        
        return K.long()
    
    def winding3(self, vdT_value, N_div, steps_per_segment, branch_cut_angle, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, plot=False, reverse=False):
        # 1. Calculate the ξ-dependent correction term for W3[Uξ].
        w3, eigvals, eigvecs = self.w3(vdT_value, N_div, steps_per_segment, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        print(w3)
        # Initialize correction_term based on branch_cut_angle type
        if isinstance(branch_cut_angle, torch.Tensor):
            correction_term = torch.zeros(len(branch_cut_angle), device=self.device)
        else:
            correction_term = 0
        # Iterate over the 2D grid at μ3 = 1 (i3 = N_div)
        for i1 in range(N_div):
            for i2 in range(N_div):
                # print(r"($i_1, i_2, i_3$)", i1+1,i2+1)
                # Compute the base point p
                p = torch.tensor([(i1) * 2*torch.pi/(N_div),
                                (i2) * 2*torch.pi/(N_div),
                                self.T], dtype=torch.float64, device=self.device)
                # print('p', p)
                # Compute F^ν_p,3
                # vertices = torch.tensor([
                #     [(i1+1)%N_div, (i2+1)%N_div, N_div],
                #     [(i1+2)%N_div, (i2+1)%N_div, N_div],
                #     [(i1+2)%N_div, (i2+2)%N_div, N_div],
                #     [(i1+1)%N_div, (i2+2)%N_div, N_div]
                # ], dtype=torch.long, device=self.device)
                vertices = torch.tensor([
                    [self.mod((i1),N_div), self.mod((i2),N_div), N_div],
                    [self.mod((i1+1),N_div), self.mod((i2),N_div), N_div],
                    [self.mod((i1+1),N_div), self.mod((i2+1),N_div), N_div],
                    [self.mod((i1),N_div), self.mod((i2+1),N_div), N_div]
                ], dtype=torch.long, device=self.device)
                # print(vertices)
                # Compute F^ν_p,3 using these vertices
                v = [eigvecs[tuple(v)] for v in vertices]
                U12 = self.U_nu(v[0], v[1])
                U23 = self.U_nu(v[1], v[2])
                U34 = self.U_nu(v[2], v[3])
                U41 = self.U_nu(v[3], v[0])
                F_p_3 = ( 1 / (2 * torch.pi * 1j)) * torch.log(U12 * U23 * U34 * U41)
                F_p_3 = F_p_3.real
                # print(F_p_3)
                # Determine K^ν_(i,j)
                K = self.determine_K(i1, i2, N_div, eigvals, eigvecs, branch_cut_angle)
                # print(K)
                # Compute the product and sum over bands
                # Handle both scalar and tensor branch_cut_angle
                if isinstance(branch_cut_angle, torch.Tensor):
                    term = torch.sum(F_p_3.unsqueeze(-1) * K, dim=0)
                else:
                    term = torch.sum(F_p_3 * K)
                
                # Add to the correction term
                correction_term += term
        print(correction_term)
        W3_U_xi = (w3 - correction_term)
        if plot:
            # Plot for multiple branch cut angles
            plt.figure(figsize=(10, 6))
            plt.plot(branch_cut_angle.cpu().numpy(), W3_U_xi.cpu().numpy(), '-o')
            plt.xlabel('Branch Cut Angle, $\\xi$')
            plt.ylabel('$W_{3}[U_{\\xi}]$')
            plt.ylim(-1.5, 1.5)
            plt.yticks(range(-1, 2))  # This sets integer ticks from -1 to 1
            # plt.title('Winding Number vs Branch Cut Angle')
            plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                [r'$-\pi$', r'$-\pi/2$','0', r'$\pi/2$', r'$\pi$'])
            plt.grid(True)
            plt.show()
        return W3_U_xi
    
    def quasienergies_states_bulk(self, steps_per_segment, vdT, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        """The quasi-energy spectrum for the bulk U(theta_x, theta_y, T) properties"""
        U = self.time_evolution_operator1(self.T, steps_per_segment, 'xy', vdT, rotation_angle, theta_x, theta_y, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        eigvals, eigvecs = torch.linalg.eig(U)
        del U
        torch.cuda.empty_cache()
        E_T = torch.log(eigvals).imag / self.T
        
        # Get the sorting indices for each batch element
        sorted_indices = torch.argsort(E_T, dim=-1)
        
        # Reorder the quasienergy, and the eigenvectors
        sorted_eigv_r = torch.gather(E_T, -1, sorted_indices)
        sorted_eigvals = torch.gather(eigvals, -1, sorted_indices)
        expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
        sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)
        
        # Free up memory
        del eigvals, eigvecs, sorted_indices, expanded_indices
        torch.cuda.empty_cache()
        
        if plot:
            # Plotting the surface
            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Ensure theta_x and theta_y are tensors
            if not isinstance(theta_x, torch.Tensor):
                theta_x = torch.tensor(theta_x, device=sorted_eigv_r.device)
            if not isinstance(theta_y, torch.Tensor):
                theta_y = torch.tensor(theta_y, device=sorted_eigv_r.device)

            X, Y = torch.meshgrid(theta_x.cpu(), theta_y.cpu(), indexing='ij')
            X_np, Y_np = X.numpy(), Y.numpy()
            sorted_eigv_r_np = sorted_eigv_r.cpu().numpy()

            for i in range(sorted_eigv_r_np.shape[-1]):
                ax.plot_surface(X_np, Y_np, sorted_eigv_r_np[:, :, i], cmap='viridis')

            ax.set_xlabel(r'$\theta_x$', fontsize=25, labelpad=20)
            ax.set_ylabel(r'$\theta_y$', fontsize=25, labelpad=20)
            ax.set_zlabel(r'Quasienergy, $T\epsilon$', fontsize=25, labelpad=5)
            ax.set_zlim(-torch.pi/self.T, torch.pi/self.T)
            ax.set_xticks([0, torch.pi, 2*torch.pi])
            ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'], fontsize=15)
            ax.set_yticks([0, torch.pi, 2*torch.pi])
            ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'], fontsize=15)
            ax.set_zticks([-torch.pi/self.T, 0, torch.pi/self.T])
            ax.set_zticklabels([r'$-\pi/T$', '0', r'$\pi/T$'], fontsize=15)
            ax.view_init(elev=1, azim=15)
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
        return sorted_eigv_r, sorted_eigvals, sorted_eigvecs

    def log_with_branchcut1(self, z, epsilonT):
        """
        Compute the logarithm with a branch cut as defined in the AFAI paper.
        
        Args:
        z (torch.Tensor): Complex tensor of eigenvalues
        epsilonT (float): Branch cut parameter times T (epsilon * T)
        
        Returns:
        torch.Tensor: Logarithm of z with the specified branch cut
        """
        # Compute chi as the phase of z
        chi = torch.angle(z)
        
        # Normalize chi to be in the range [0, 2π)
        chi = chi % (2 * torch.pi)
        
        # Adjust chi based on the branch cut definition
        chi_adjusted = torch.where(
            chi < epsilonT,
            chi,
            chi - 2 * torch.pi
        )
        
        return chi_adjusted * 1j
    
    def H_eff(self, steps_per_segment, vdT, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT = torch.pi, delta=None, initialise=False, fully_disorder=True):
            delete, eigenvalues_matrix, wf_matrix = self.quasienergies_states_bulk(steps_per_segment, vdT, theta_x, theta_y, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
            del delete
            log_eigenvalues = self.log_with_branchcut1(eigenvalues_matrix, epsilonT)
            # Initialize H_eff with the same shape and device as wf_matrix
            # Multiply by (1j / self.T) here
            log_eigenvalues = (1j / self.T) * log_eigenvalues

            H_eff = torch.zeros_like(wf_matrix, dtype=torch.complex128)

            # Reshape eigenvalues_matrix to (theta_x*theta_y, size) and convert to complex
            eigenvalues_flat = log_eigenvalues.reshape(-1, log_eigenvalues.shape[-1]).to(torch.complex128)

            # Create diagonal matrices for all theta-points at once
            H_diag = torch.diag_embed(eigenvalues_flat)

            # Reshape wf_matrix to (theta_x*theta_y, size, size)
            wf_flat = wf_matrix.reshape(-1, wf_matrix.shape[-2], wf_matrix.shape[-1])

            # Ensure wf_flat is complex
            wf_flat = wf_flat.to(torch.complex128)

            # Compute H_eff for all k-points in one batch operation
            H_eff_flat = torch.bmm(torch.bmm(wf_flat, H_diag), wf_flat.conj().transpose(-2, -1))

            # Reshape H_eff back to original shape
            H_eff = H_eff_flat.reshape_as(wf_matrix)

            return H_eff, log_eigenvalues, wf_matrix
    
    def get_theta_values(self, theta_num, tbc):
        if tbc == 'x':
            theta_x = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
            theta_y = torch.zeros(1, device=self.device)
        elif tbc == 'y':
            theta_x = torch.zeros(1, device=self.device)
            theta_y = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
        elif tbc == 'xy':
            theta_x = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
            theta_y = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
        return theta_x, theta_y
    
    ### Now since the shape of U: (theta_x, theta_y, t, size, size) are so large that the current GPU memory is not enough for a larger system.
    def compute_deformed_U(self, t_num, theta_num, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT = torch.pi, delta=None, initialise=False, fully_disorder=True):
        """Here we take single vdT value due to the large tensor size. Therefore, the function does not support vdT to be a 1D tensor though the U_t may have the shape (vdT, thetax, thetay, t, size, size)"""
        # Generate t values
        t_num += 1
        t = torch.linspace(0, self.T, t_num, device=self.device)

        # First, compute U(thetax, thetay, t) for 0 <= t <= T
        thetax, thetay = self.get_theta_values(theta_num+1, 'xy')
        U_t = self.time_evolution_operator1(t, steps_per_segment, 'xy', vdT, rotation_angle, thetax, thetay, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        
        # Compute H_eff and get log_eigenvalues and S (eigenvectors)
        H_eff, log_eigenvalues, S = self.H_eff(steps_per_segment,vdT, thetax, thetay, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        del H_eff
        # Reshape t for broadcasting
        t_reshaped = t.reshape(1, 1, -1, 1)
    
        # Compute exp(1j*t*log_eigenvalues) for all time steps
        exp_term = torch.exp(1j * t_reshaped * log_eigenvalues.unsqueeze(2))
        
        # Reshape exp_term for matrix multiplication
        exp_term_diag = torch.diag_embed(exp_term)
        
        # Expand S to match the time dimension
        S_expanded = S.unsqueeze(2).expand(-1, -1, t_num, -1, -1)
        
        # Compute S * exp(1j*t*log_eigenvalues) * S^+
        deformation_factor = torch.einsum('...tij,...tjk,...tkl->...til', S_expanded, exp_term_diag, S_expanded.conj().transpose(-1, -2))
        
        # Compute the deformed U for all time steps
        U_prime = torch.einsum('...tij,...tjk->...tik', U_t, deformation_factor)
        
        return U_prime
        
    def commutator(self, A, B):
        return torch.matmul(A, B) - torch.matmul(B, A)
    
    def compute_U_derivatives_central(self, U):
        """
            Compute the derivatives of U with respect to kx, ky, and t using central differences.

        Args:
        U (torch.Tensor): The U tensor of shape (theta_num+1, theta_num+1, theta_num+1, size, size)

        Returns:
        tuple: (dU_dthetax, dU_dthetay, dU_dt) with shapes:
            dU_dthetax: (theta_num-1, theta_num+1, t_num+1, size, size)
            dU_dthetay: (theta_num+1, theta_num-1, t_num+1, size, size)
            dU_dt: (theta_num+1, theta_num+1, t_num-1, size, size)
        """
        theta_num, _, t_num, size, _ = U.shape
        theta_num -= 1
        t_num -= 1
        # Compute step sizes
        dthetax = (2 * torch.pi) / theta_num
        # print('dthetax',dthetax)
        dthetay = (2 * torch.pi) / theta_num
        # print('dthetax',dthetax)
        dt = self.T / t_num
        # print('dt', dt)
        # Compute central differences for derivatives
        dU_dthetax = (U[2:, :, :, :, :] - U[:-2, :, :, :, :]) / (2 * dthetax)
        dU_dthetay = (U[:, 2:, :, :, :] - U[:, :-2, :, :, :]) / (2 * dthetay)
        dU_dt = (U[:, :, 2:, :, :] - U[:, :, :-2, :, :]) / (2 * dt)
        # Debug prints before trimming
        # print('Before trimming:')
        # print('shape of dU_dthetax', dU_dthetax.shape)  # Shape: (theta_num-1, theta_num+1, t_num+1, size, size)
        # print('shape of dU_dthetay', dU_dthetay.shape)  # Shape: (theta_num+1, theta_num-1, t_num+1, size, size)
        # print('shape of dU_dt', dU_dt.shape)    # Shape: (theta_num+1, theta_num+1, t_num-1, size, size)

        # Trim the boundaries to ensure consistent shape
        dU_dthetax = dU_dthetax[:, 1:-1, 1:-1, :, :]
        dU_dthetay = dU_dthetay[1:-1, :, 1:-1, :, :]
        dU_dt = dU_dt[1:-1, 1:-1, :, :, :]
        return dU_dthetax, dU_dthetay, dU_dt
        
    def compute_integrand(self, U):
        """
        Compute the integrand for the triple integral in formula 8 using central differences.

        Args:
        U (torch.Tensor): The U tensor of shape (theta_num+1, theta_num+1, t_num+1, size, size)

        Returns:
        torch.Tensor: The integrand of shape (theta_num-2, theta_num-2, t_num-2)
        """
        dU_dthetax, dU_dthetay, dU_dt = self.compute_U_derivatives_central(U)

        # Trim U to match the shape of derivatives
        U = U[1:-1, 1:-1, 1:-1, :, :]
        U_dag = U.conj().transpose(-2, -1)
        # print('the shape of trimmed U', U.shape)
        term1 = torch.matmul(U_dag, dU_dt)
        # print('the shape of term1', term1.shape)
        term2 = self.commutator(torch.matmul(U_dag, dU_dthetax), torch.matmul(U_dag, dU_dthetay))
        # print('the shape of term2', term2.shape)
        integrand = torch.einsum('...ii', torch.matmul(term1, term2))
        # print('integrand', integrand.shape)
        real_integrand = integrand.real
        # print(integrand[0,0])
        return real_integrand

    def compute_winding_number(self, U):
        """
        Compute the winding number W3 using the central difference method for the triple integral.

        Args:
        U (torch.Tensor): The U tensor of shape (theta_num+1, theta_num+1, t_num+1, size, size)

        Returns:
        float: The computed winding number.
        """
        integrand = self.compute_integrand(U)
        
        # Get the dimensions of the integrand
        theta_num, _, t_num, size, _ = U.shape
        theta_num -= 1
        t_num -= 1

        # Compute step sizes
        dthetax = (2 * torch.pi) / theta_num
        # print('dthetax',dthetax)
        dthetay = (2 * torch.pi) / theta_num
        # print('dthetay',dthetay)
        dt = self.T / t_num
        # print('dt', dt)

        # Apply central difference integration
        integral_sum = torch.sum(integrand)
        winding_number = (1 / (8 * torch.pi**2)) * integral_sum * dthetax * dthetay * dt

        return winding_number
        
    def plot_W3_vs_xi(self, theta_num, t_num, steps_per_segment, N_xi, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT = torch.pi, delta=None, initialise=False, fully_disorder=True):
        """
        Generate a plot of W₃[Uξ] versus ξ for different branch cut values for single realisation (no matter disorder or phase).

        Args:
        theta_num (int): Number of theta-points in each direction.
        t_num (int): Number of time steps.
        steps_per_segment (int): Number of steps per segment in the time evolution.
        N_xi (int): Number of ξ values to compute.
        delta (float, optional): Delta parameter for time evolution.

        Returns:
        None (displays the plot)
        """
        # Generate ξ values
        xi_values = torch.linspace(0, 2*torch.pi, N_xi, device=self.device)
        
        # Initialize W3_values as a zero tensor
        W3_values = torch.zeros(N_xi, device=self.device)

        for i, xi in enumerate(xi_values):
            # Compute the deformed U for this ξ value
            U_xi = self.compute_deformed_U(t_num, theta_num, steps_per_segment, vdT, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
            
            # Compute W₃[Uξ] for this ξ value and store it directly in W3_values
            W3_values[i] = self.compute_winding_number(U_xi)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(xi_values.cpu().numpy(), W3_values.cpu().numpy(), '-o')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$W_3[U_\xi]$')
        plt.ylim(-1.5, 1.5)
        plt.yticks(range(-1, 2))  # This sets integer ticks from -1 to 1
        # plt.title(r'Winding Number $W_3[U_\xi]$ vs $\xi$')
        plt.grid(True)
        # plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference
        
        # Set x-axis ticks to multiples of π
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

        plt.show()
        
    def plot_W3_vs_xi_disorder_averaged(self, theta_num, t_num, steps_per_segment, N_xi, N_dis, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False):
        """
        Generate a plot of disorder-averaged W₃[Uξ] versus ξ for different branch cut values.
        """
        xi_values = torch.linspace(0, 2*torch.pi, N_xi, device=self.device)
        W3_values = torch.zeros(N_xi, device=self.device)

        for i, xi in enumerate(xi_values):
            W3_sum = 0
            for _ in range(N_dis):
                self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
                gc.collect()  # Force garbage collection
                torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                
                U_xi = self.compute_deformed_U(t_num, theta_num, steps_per_segment, vdT, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=True)
                
                W3 = self.compute_winding_number(U_xi)
                W3_sum += W3
                
                del U_xi, W3  # Delete variables to free memory
                gc.collect()  # Force garbage collection
                torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
            
            W3_values[i] = W3_sum / N_dis
            print(f"Completed ξ value {i+1}/{N_xi}")  # Progress indicator

        # Move data to CPU for plotting
        xi_values_cpu = xi_values.cpu().numpy()
        W3_values_cpu = W3_values.cpu().numpy()
        
        # Clear GPU memory
        del xi_values, W3_values
        torch.cuda.empty_cache()

        plt.figure(figsize=(10, 6))
        plt.plot(xi_values_cpu, W3_values_cpu, '-o')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\langle W_3[U_\xi] \rangle$')
        # plt.title(r'Disorder-Averaged Winding Number $\langle W_3[U_\xi] \rangle$ vs $\xi$ (N_dis = {N_dis})')
        plt.grid(True)
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.show()

        # Clear remaining variables
        del xi_values_cpu, W3_values_cpu
        gc.collect()
        torch.cuda.empty_cache()
    
    def plot_W3_vs_vdT(self, theta_num, t_num, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        """
        Generate a plot of W₃[Uξ] versus the aperiodic strength, vdT where only two branch cut values ξ=0 and pi are chosen.
        """
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        W3_values = torch.zeros((len(vdT), len(xi_values)), device=self.device)
        for i, aperiodic_strength in enumerate(vdT):
            print(f"Computing vdT value {i+1}/{len(vdT)}")
            for j, xi in enumerate(xi_values):
                # Compute the deformed U for this ξ value
                U_xi = self.compute_deformed_U(t_num, theta_num, steps_per_segment, aperiodic_strength, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
                # Compute W₃[Uξ] for this ξ value and store it directly in W3_values
                W3_values[i,j] = self.compute_winding_number(U_xi)
                del U_xi
                torch.cuda.empty_cache()
        # Convert tensors to numpy arrays for plotting
        vdT_np = vdT.cpu().numpy()
        W3_values_np = W3_values.cpu().numpy()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(vdT_np, W3_values_np[:, 0], label='ξ = 0', marker='o')
        plt.plot(vdT_np, W3_values_np[:, 1], label='ξ = π', marker='s')
        plt.ylim(-1.5, 1.5)
        plt.yticks(range(-1, 2))  # This sets integer ticks from -1 to 1
        plt.xlabel('Aperiodic Strength (vdT)')
        plt.ylabel('Winding Number W₃[Uξ]')
        plt.legend()
        plt.grid(True)
        plt.show()
        return W3_values
    
    def plot_W3_vdT_disorder_realisation(self, theta_num, t_num, steps_per_segment, vdT, N_dis, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False):
        """
        Generate a plot of disorder-averaged W₃[Uξ] versus the aperiodic strength, vdT where only two branch cut values ξ=0 and π are chosen.
        """
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        W3_values = torch.zeros((len(vdT), len(xi_values)), device=self.device)
        
        for i, aperiodic_strength in enumerate(vdT):
            for j, xi in enumerate(xi_values):
                W3_sum = 0
                for _ in range(N_dis):
                    self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
                    gc.collect()  # Force garbage collection
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                    
                    # Compute the deformed U for this ξ value and disorder realization
                    U_xi = self.compute_deformed_U(t_num, theta_num, steps_per_segment, aperiodic_strength, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=True)
                    
                    # Compute W₃[Uξ] for this ξ value
                    W3 = self.compute_winding_number(U_xi)
                    W3_sum += W3
                    
                    del U_xi, W3  # Delete variables to free memory
                    gc.collect()  # Force garbage collection
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                    
                # Average over disorder realizations
                W3_values[i, j] = W3_sum / N_dis
                print(f"Completed vdT value {i+1}/{len(vdT)}, ξ = {'0' if j == 0 else 'π'}")  # Progress indicator
        
        # Convert tensors to numpy arrays for plotting
        vdT_np = vdT.cpu().numpy()
        W3_values_np = W3_values.cpu().numpy()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(vdT_np, W3_values_np[:, 0], label='ξ = 0', marker='o')
        plt.plot(vdT_np, W3_values_np[:, 1], label='ξ = π', marker='s')
        plt.ylim(-1.5, 1.5)
        plt.yticks(range(-1, 2))  # This sets integer ticks from -1 to 1
        plt.xlabel('Aperiodic Strength (vdT)')
        plt.ylabel('Winding Number W₃[Uξ]')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return W3_values
    
    ## Now rewrite the above few function without batch processing but calculate every scalar thetax, thetay and t values. (Take very long time.....)
    def compute_deformed_U_single(self, t, theta_x, theta_y, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        """
        Compute the deformed U for a single set of t, theta_x, and theta_y values.
        """
        U_t = self.time_evolution_operator1(t, steps_per_segment, 'xy', vdT, rotation_angle, theta_x, theta_y, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        
        H_eff, log_eigenvalues, S = self.H_eff(steps_per_segment, vdT, theta_x, theta_y, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        del H_eff
        exp_term = torch.exp(1j * t * log_eigenvalues)
        deformation_factor = S @ torch.diag(exp_term) @ S.conj().T
        
        U_prime = U_t @ deformation_factor
        
        return U_prime
    
    def compute_U_derivatives_central_single(self, theta_x, theta_y, t, d_theta, d_t, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        """
        Compute the derivatives of U with respect to theta_x, theta_y, and t using central differences for a single point.
        """
        # Compute U at the central point
        U_center = self.compute_deformed_U_single(t, theta_x, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)

        # Compute derivatives with respect to theta_x
        U_plus_x = self.compute_deformed_U_single(t, theta_x + d_theta, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        U_minus_x = self.compute_deformed_U_single(t, theta_x - d_theta, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        dU_dthetax = (U_plus_x - U_minus_x) / (2 * d_theta)

        # Compute derivatives with respect to theta_y
        U_plus_y = self.compute_deformed_U_single(t, theta_x, theta_y + d_theta, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        U_minus_y = self.compute_deformed_U_single(t, theta_x, theta_y - d_theta, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        dU_dthetay = (U_plus_y - U_minus_y) / (2 * d_theta)

        # Compute derivatives with respect to t
        U_plus_t = self.compute_deformed_U_single(t + d_t, theta_x, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        U_minus_t = self.compute_deformed_U_single(t - d_t, theta_x, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        dU_dt = (U_plus_t - U_minus_t) / (2 * d_t)

        return U_center, dU_dthetax, dU_dthetay, dU_dt
    
    def compute_winding_number_single(self, theta_num, t_num, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        d_theta = 2 * torch.pi / theta_num
        d_t = self.T / t_num
        
        integral_sum = 0
        for i in range(1, theta_num - 1):  # Start from 1, end at theta_num - 1
            for j in range(1, theta_num - 1):  # Start from 1, end at theta_num - 1
                for k in range(1, t_num - 1):  # Start from 1, end at t_num - 1
                    theta_x = 2 * torch.pi * i / theta_num
                    theta_y = 2 * torch.pi * j / theta_num
                    t = self.T * k / t_num
                    
                    U, dU_dthetax, dU_dthetay, dU_dt = self.compute_U_derivatives_central_single(theta_x, theta_y, t, d_theta, d_t, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                    
                    U_dag = U.conj().T
                    term1 = torch.matmul(U_dag, dU_dt)
                    term2 = self.commutator(torch.matmul(U_dag, dU_dthetax), torch.matmul(U_dag, dU_dthetay))
                    integrand = torch.trace(torch.matmul(term1, term2)).real
                    
                    integral_sum += integrand

        winding_number = (1 / (8 * torch.pi**2)) * integral_sum * d_theta * d_theta * d_t

        return winding_number
    
    def plot_W3_vs_xi_single(self, theta_num, t_num, steps_per_segment, N_xi, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        """
        Generate a plot of W₃[Uξ] versus ξ for different branch cut values for single realisation (no matter disorder or phase).

        Args:
        theta_num (int): Number of theta-points in each direction.
        t_num (int): Number of time steps.
        steps_per_segment (int): Number of steps per segment in the time evolution.
        N_xi (int): Number of ξ values to compute.
        vdT (float): Disorder strength parameter.
        a, b, phi1_ex, phi2_ex (float): System parameters.
        rotation_angle (float): Rotation angle parameter.
        delta (float, optional): Delta parameter for time evolution.
        initialise (bool): Whether to initialize the system.
        fully_disorder (bool): Whether to use full disorder.

        Returns:
        None (displays the plot)
        """
        # Generate ξ values
        xi_values = torch.linspace(0, 2*torch.pi, N_xi, device=self.device)
        
        # Initialize W3_values as a zero tensor
        W3_values = torch.zeros(N_xi, device=self.device)

        for i, xi in enumerate(xi_values):
            # Compute W₃[Uξ] for this ξ value and store it directly in W3_values
            W3_values[i] = self.compute_winding_number_single(theta_num, t_num, steps_per_segment, vdT, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=fully_disorder)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(xi_values.cpu().numpy(), W3_values.cpu().numpy(), '-o')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$W_3[U_\xi]$')
        plt.grid(True)
        
        # Set x-axis ticks to multiples of π
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

        plt.show()
    
    ## Maybe batch along one direction or two directions?
    def compute_deformed_U_batch(self, t_batch, theta_x_batch, theta_y_batch, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        # print('t batch', t_batch.shape)
        # print('theta_x',theta_x_batch.shape)
        # print('theta_y',theta_y_batch.shape)
        U_t = self.time_evolution_operator1(t_batch, steps_per_segment, 'xy', vdT, rotation_angle, theta_x_batch, theta_y_batch, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        # print('U_t', U_t.shape)
        H_eff, log_eigenvalues, S = self.H_eff(steps_per_segment, vdT, theta_x_batch, theta_y_batch, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        del H_eff
        # print('H_eff', H_eff.shape)
        # print('S', S.shape)
        # print('log', log_eigenvalues.shape)
        exp_term = torch.exp(1j * t_batch * log_eigenvalues)
        deformation_factor = torch.bmm(torch.bmm(S, torch.diag_embed(exp_term)), S.conj().transpose(-2, -1))
        
        U_prime = torch.bmm(U_t, deformation_factor)
        
        return U_prime

    def compute_winding_number_batchx(self, theta_num, t_num, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        '''Batch along thetax direction only. Too slow'''
        d_theta = 2 * torch.pi / theta_num
        d_t = self.T / t_num
        
        # Generate all theta_x values at once
        theta_x_all = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
        
        integral_sum = 0
        for j in range(1, theta_num - 1):
            theta_y = 2 * torch.pi * j / theta_num
            theta_y_plus = 2 * torch.pi * (j + 1) / theta_num
            theta_y_minus = 2 * torch.pi * (j - 1) / theta_num
            
            for k in range(1, t_num - 1):
                print(j,k)
                t = self.T * k / t_num
                t_plus = self.T * (k + 1) / t_num
                t_minus = self.T * (k - 1) / t_num
                
                U = self.compute_deformed_U_batch(t, theta_x_all, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                
                # Compute dU_dthetax directly
                dU_dthetax = (U[2:, :, :] - U[:-2, :, :]) / (2 * d_theta)
                
                # Compute dU_dthetay
                U_plus_y = self.compute_deformed_U_batch(t, theta_x_all, theta_y_plus, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                U_minus_y = self.compute_deformed_U_batch(t, theta_x_all, theta_y_minus, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                dU_dthetay = (U_plus_y - U_minus_y) / (2 * d_theta)
                
                # Compute dU_dt
                U_plus_t = self.compute_deformed_U_batch(t_plus, theta_x_all, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                U_minus_t = self.compute_deformed_U_batch(t_minus, theta_x_all, theta_y, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
                dU_dt = (U_plus_t - U_minus_t) / (2 * d_t)
                
                # Adjust U and U_dag to match the shape of dU_dthetax
                U = U[1:-1, :, :]
                U_dag = U.conj().transpose(-2, -1)
                
                term1 = torch.bmm(U_dag, dU_dt[1:-1, :, :])
                term2 = self.commutator(torch.bmm(U_dag, dU_dthetax), torch.bmm(U_dag, dU_dthetay[1:-1, :, :]))
                integrand = torch.einsum('bii->b', torch.bmm(term1, term2)).real
                
                integral_sum += integrand.sum()
            
            torch.cuda.empty_cache()

        winding_number = (1 / (8 * torch.pi**2)) * integral_sum * d_theta * d_theta * d_t
        return winding_number

    def compute_winding_number_batch(self, theta_num, t_num, steps_per_segment, vdT, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        '''Batch along both thetax and thetay dimensions'''
        d_theta = 2 * torch.pi / theta_num
        d_t = self.T / t_num
        
        # Generate all theta_x and theta_y values at once
        theta_values = torch.linspace(0, 2*torch.pi, theta_num, device=self.device)
        
        integral_sum = 0
        for k in range(1, t_num - 1):
            print(f"Time step: {k}/{t_num-2}")
            t = self.T * k / t_num
            t_plus = self.T * (k + 1) / t_num
            t_minus = self.T * (k - 1) / t_num
            
            # Compute U for all theta_x and theta_y combinations
            U = self.compute_deformed_U_batch(t, theta_values, theta_values, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
            
            # Compute dU_dthetax directly
            dU_dthetax = (U[2:, :, :, :] - U[:-2, :, :, :]) / (2 * d_theta)
            
            # Compute dU_dthetay directly
            dU_dthetay = (U[:, 2:, :, :] - U[:, :-2, :, :]) / (2 * d_theta)
            
            # Compute dU_dt
            U_plus_t = self.compute_deformed_U_batch(t_plus, theta_values, theta_values, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
            U_minus_t = self.compute_deformed_U_batch(t_minus, theta_values, theta_values, steps_per_segment, vdT, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
            dU_dt = (U_plus_t - U_minus_t) / (2 * d_t)
            
            # Adjust U and U_dag to match the shape of dU_dthetax and dU_dthetay
            U = U[1:-1, 1:-1, :, :]
            U_dag = U.conj().transpose(-2, -1)
            
            term1 = torch.einsum('abij,abjk->abik', U_dag, dU_dt[1:-1, 1:-1, :, :])
            term2 = self.commutator(
                torch.einsum('abij,abjk->abik', U_dag, dU_dthetax[:, 1:-1, :, :]),
                torch.einsum('abij,abjk->abik', U_dag, dU_dthetay[1:-1, :, :, :])
            )
            integrand = torch.einsum('abii->ab', torch.einsum('abij,abjk->abik', term1, term2)).real
            integral_sum += integrand.sum()
            
            del U, U_plus_t, U_minus_t, dU_dthetax, dU_dthetay, dU_dt, U_dag, term1, term2, integrand
            torch.cuda.empty_cache()

        winding_number = (1 / (8 * torch.pi**2)) * integral_sum * d_theta * d_theta * d_t
        return winding_number
    
    def plot_W3_vs_xi_batch(self, theta_num, t_num, steps_per_segment, N_xi, vdT, batch_size=10, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        xi_values = torch.linspace(0, 2*torch.pi, N_xi, device=self.device)
        W3_values = torch.zeros(N_xi, device=self.device)

        for i, xi in enumerate(xi_values):
            W3_values[i] = self.compute_winding_number_batch(theta_num=theta_num, t_num=t_num, steps_per_segment=steps_per_segment, vdT=vdT, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, epsilonT=xi, delta=delta, initialise=initialise, fully_disorder=fully_disorder)

        plt.figure(figsize=(10, 6))
        plt.plot(xi_values.cpu().numpy(), W3_values.cpu().numpy(), '-o')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$W_3[U_\xi]$')
        plt.grid(True)
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                   ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.show()
    
    def avg_level_spacing_bulk(self, steps_per_segment, vdT, theta_x=0, theta_y=0, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The level spacing statistics of the bulk evolution operator for a batch of vd values'''
        # Ensure vdT is a tensor on the correct device
        if not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)

        # Ensure other parameters are on the correct device
        theta_x = torch.tensor(theta_x, device=self.device) if not isinstance(theta_x, torch.Tensor) else theta_x.to(self.device)
        theta_y = torch.tensor(theta_y, device=self.device) if not isinstance(theta_y, torch.Tensor) else theta_y.to(self.device)
        rotation_angle = torch.tensor(rotation_angle, device=self.device) if not isinstance(rotation_angle, torch.Tensor) else rotation_angle.to(self.device)

        # Use no_grad for memory efficiency during inference
        with torch.no_grad():
            # Clear cache before starting
            torch.cuda.empty_cache()

            E_T, eigvals, _ = self.quasienergies_states_bulk(steps_per_segment, vdT, theta_x, theta_y, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
            # Compute level spacing for each batch element
            difff = torch.diff(E_T, dim=1)
            level_spacing = torch.minimum(difff[:, 1:], difff[:, :-1]) / torch.maximum(difff[:, 1:], difff[:, :-1])
            level_spacing_avg = level_spacing.mean(dim=1)

            # Explicitly delete large tensors and clear cache
            del E_T, _, difff, level_spacing, eigvals
            torch.cuda.empty_cache()

        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            tick_label_fontsize = 32
            label_fontsize = 34
            x_vals = vdT.squeeze()
            ax.scatter(x_vals.cpu().numpy(), level_spacing_avg.cpu().numpy(), c='b')
            ax.set_xlabel(r'Aperiodic potential, $\delta V$T', fontsize=label_fontsize)
            ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=label_fontsize)
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

        return level_spacing_avg
    
    def avg_LSR_disorder_realisation(self, steps_per_segment, vdT_min, vdT_max, vdT_num, N_dis, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of disorder realisation"""
        vdT = torch.linspace(vdT_min, vdT_max, vdT_num, device=self.device)
        avg = torch.zeros(vdT_num, device=self.device)

        for _ in range(N_dis):
            print("disorder configuration number:",_)
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            avg_single = self.avg_level_spacing_bulk(steps_per_segment, vdT, delta=delta, fully_disorder=True)
            avg += avg_single
            torch.cuda.empty_cache()

        avg_LSR = avg / N_dis
        avg_LSR_cpu = avg_LSR.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd_plot = vdT.cpu().numpy()
        ax.scatter(vd_plot, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Disorder strength, $\delta V$T', fontsize=label_fontsize)
        ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.axhline(y=0.386, color='r', linestyle='--', label='Poisson')
        ax.axhline(y=0.5996, color='g', linestyle='--', label='GUE')
        ax.axhline(y=0.53, color='black', linestyle='--', label='GOE')
        ax.legend(fontsize=tick_label_fontsize, loc='best')

        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
        return avg_LSR
    
    def avg_LSR_phase_realisation(self, steps_per_segment, vdT_min, vdT_max, vdT_num, N_phi, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of phase realisation"""
        vdT = torch.linspace(vdT_min, vdT_max, vdT_num, device=self.device)
        avg = torch.zeros(vdT_num, device=self.device)
        phi1_vals = torch.rand(N_phi, device=self.device) * 2 * np.pi
        phi2_vals = torch.rand(N_phi, device=self.device) * 2 * np.pi

        for i in range(N_phi):
            avg_single = self.avg_level_spacing_bulk(steps_per_segment, vdT, phi1_ex=phi1_vals[i], phi2_ex=phi2_vals[i], 
                                                    delta=delta, fully_disorder=False)
            avg += avg_single
            torch.cuda.empty_cache()

        avg_LSR = avg / N_phi
        avg_LSR_cpu = avg_LSR.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd_plot = vdT.cpu().numpy()
        ax.scatter(vd_plot, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Quasiperiodic potential strength, $\delta V$T', fontsize=label_fontsize)
        ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.axhline(y=0.386, color='r', linestyle='--', label='Poisson')
        ax.axhline(y=0.53, color='black', linestyle='--', label='GOE')
        ax.axhline(y=0.5996, color='g', linestyle='--', label='GUE')
        ax.legend(fontsize=tick_label_fontsize, loc='best')

        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
        return avg_LSR
    
    ## Exploring the edge properties of the system
    ## Function 1. Quasienergies and states -- COMPLETED
    ## Function 2. Deformed time-periodic evolution operator --COMPLETED
    ## Function 3. Edge state invariant --COMPLETED
    ## Function 4. Disordered-averaged transmission probability --COMPLETED
    ## Function 5. Quantised Charge Pumping --COMPLETED
    ## Function 6. The Inverse Participation Ratios
    
    def quasienergies_states_edge(self, steps_per_segment, tbc, vdT, theta_p_num, rotation_angle = torch.pi/4, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The quasi-energy spectrum for the edge U(kx, T) or U(ky, T) properties'''
        '''The output should be the quasienergies which are the diagonal elements of the effective Hamiltonian after diagonalisation. This is an intermediate step towards the 'deformed' time-periodic evolution operator '''
        size = self.nx * self.ny
        theta_p_num += 1
        # Ensure vd is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        else:
            vdT = vdT.to(self.device)
        vdT = vdT.reshape(-1, 1, 1)  # Reshape for broadcasting
        
        if tbc == 'x':
            theta = torch.linspace(0, 2 * torch.pi, theta_p_num, device=self.device, dtype=torch.float64)
            theta_fixed = torch.tensor([0], device=self.device, dtype=torch.float64)
            U = self.time_evolution_operator1(self.T, steps_per_segment, tbc, vdT, rotation_angle, theta, theta_fixed, a, b, phi1, phi2, delta, initialise, fully_disorder)
        elif tbc == 'y':
            theta = torch.linspace(0, 2 * torch.pi, theta_p_num, device=self.device, dtype=torch.float64)
            theta_fixed = torch.tensor([0], device=self.device, dtype=torch.float64)
            U = self.time_evolution_operator1(self.T, steps_per_segment, tbc, vdT, rotation_angle, theta_fixed, theta, a, b, phi1, phi2, delta, initialise, fully_disorder)
        else:
            raise ValueError("tbc must be either 'x' or 'y'")

        eigvals, eigvecs = torch.linalg.eig(U)
        # print("Original eigvecs shape:", eigvecs.shape)
        # print("Original eigvecs norm:", torch.norm(eigvecs, dim=-2))
        # print(eigvecs[0])  # Print the first matrix of eigenvectors
        del U
        E_T = 1j * torch.log(eigvals) / self.T
        E_T_real = E_T.real

        # Sort the real parts of E_T
        sorted_indices = torch.argsort(E_T_real, dim=-1)

        # Reorder the quasienergy, and the eigenvectors
        sorted_eigv_r = torch.gather(E_T, -1, sorted_indices)
        sorted_eigvals = torch.gather(eigvals, -1, sorted_indices)
        expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
        sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)

        # print(sorted_eigvecs[0])  # Print the first matrix of sorted eigenvectors
        # print("Sorted eigvecs shape:", sorted_eigvecs.shape)
        # print("Sorted eigvecs norm:", torch.norm(sorted_eigvecs, dim=-2))

        # Check orthogonality
        # dot_products = torch.matmul(sorted_eigvecs.transpose(-1, -2).conj(), sorted_eigvecs)
        # off_diagonal = dot_products - torch.eye(dot_products.shape[-1], device=dot_products.device)
        # print("Max off-diagonal element:", torch.max(torch.abs(off_diagonal)))

        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            tick_label_fontsize = 32
            label_fontsize = 34

            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=tick_label_fontsize)

            theta_cpu = theta.cpu().numpy()
            if sorted_eigv_r.dim() == 3:
                eigenvalues_matrix_cpu = sorted_eigv_r[0].cpu().numpy()
            elif sorted_eigv_r.dim() == 2:
                eigenvalues_matrix_cpu = sorted_eigv_r.cpu().numpy()
            # print(f"Shape of theta_{tbc}_cpu:", theta_cpu.shape)
            # print(f"Values of theta_{tbc}_cpu:", theta_cpu)
            # print("Shape of eigenvalues_matrix_cpu:", eigenvalues_matrix_cpu.shape)
            ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
            ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
            
            # Plot for the first vd in the batch
            for i in range(theta_p_num):
                # print(f"Index i: {i}")
                # print("Scatter plot data length:", len(eigenvalues_matrix_cpu[i]))
                ax.scatter([theta_cpu[i]] * size, eigenvalues_matrix_cpu[i].real, c='b', s=0.1)

            ax.set_xlabel(rf'$\theta_{tbc}$', fontsize=label_fontsize)
            ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(-np.pi / self.T, np.pi / self.T)

            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

        return sorted_eigv_r, sorted_eigvals, sorted_eigvecs
    
    def plot_dos(self, steps_per_segment, tbc, vdT, theta_p_num, rotation_angle = torch.pi/4, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True, num_bins=50, save_path=None):
        sorted_eigv_r, _, _ = self.quasienergies_states_edge(steps_per_segment, tbc, vdT, theta_p_num, rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        quasienergies = sorted_eigv_r.cpu().numpy()
        flat_quasienergies = np.real(quasienergies).flatten()
        # Calculate the histogram
        hist, bin_edges = np.histogram(flat_quasienergies, bins=num_bins, range=(-np.pi/self.T, np.pi/self.T))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.plot(hist, bin_centers, 'b-')
        ax.fill_betweenx(bin_centers, 0, hist, alpha=0.3)
        
        # Set labels and ticks
        ax.set_ylabel(r'Quasienergy, $\varepsilon$', fontsize=14)
        ax.set_xlabel('DOS', fontsize=14)
        ax.set_ylim(-np.pi/self.T, np.pi/self.T)
        ax.set_yticks([-np.pi/self.T, -np.pi/(2*self.T), 0, np.pi/(2*self.T), np.pi/self.T])
        ax.set_yticklabels([r'$-\frac{\pi}{T}$', r'$-\frac{\pi}{2T}$', '0', r'$\frac{\pi}{2T}$', r'$\frac{\pi}{T}$'])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
    
    def eigenstate_prob_density(self, wave_f, figsize, quasi=False, scale_factor=20, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, grid_density=100, fix_z_scale=True, save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        tick_label_fontsize = 32
        label_fontsize = 34
        
        # Calculate probability density
        prob_density = (wave_f.abs() ** 2).cpu().numpy().reshape(self.ny, self.nx)
        scaled_prob_d = prob_density * scale_factor
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y)
        
        ls = LightSource(azdeg=315, altdeg=45)
        
        # Use the custom colormap
        cmap = LinearSegmentedColormap.from_list('white_purple', ['white', 'purple'], N=100)
        
        surf = ax.plot_surface(X, Y, scaled_prob_d, 
                            cmap=cmap,
                            rstride=1, cstride=1,
                            linewidth=0,
                            antialiased=False,
                            shade=True, lightsource=ls)
        
        if quasi:
            # Generate quasiperiodic potential
            rotation_angle = rotation_angle.to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)
            x_dense = np.linspace(-1, self.nx, grid_density) - (self.nx - 1) / 2
            y_dense = np.linspace(-1, self.ny, grid_density) - (self.ny - 1) / 2
            X_dense, Y_dense = np.meshgrid(x_dense + (self.nx - 1) / 2, y_dense + (self.ny - 1) / 2, indexing='ij')
            Z = self.quasip_continuum(x_dense, y_dense, a, b, phi1_ex, phi2_ex, rotation_angle)

            # Add quasiperiodic contour plot at the bottom
            offset = -np.max(scaled_prob_d) * 1.1  # Set the offset to 10% below the minimum of the probability density
            quasi_contour = ax.contourf(X_dense, Y_dense, Z, levels=100, zdir='z', offset=offset, cmap='viridis', alpha=0.7)
            
            # Add colorbar for quasiperiodic potential
            cbar_quasi = fig.colorbar(quasi_contour, ax=ax, shrink=0.5, aspect=5, pad=0.15)
            cbar_quasi.set_label('Quasiperiodic Potential', fontsize=label_fontsize, labelpad=20)
            cbar_quasi.ax.tick_params(labelsize=tick_label_fontsize)

        if fix_z_scale:
            if quasi:
                ax.set_zlim((offset, np.max(scaled_prob_d)))
            else:
                ax.set_zlim((0, np.max(scaled_prob_d)))
        
        ax.set_xlabel('X', labelpad=100, fontsize=label_fontsize)
        ax.set_ylabel('Y', labelpad=100, fontsize=label_fontsize)
        
        # Set 5 equally spaced tick labels for x and y axes
        x_ticks = np.linspace(0, self.nx-1, 5, dtype=int)
        y_ticks = np.linspace(0, self.ny-1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        
        # Remove z-axis tick labels
        ax.set_zticklabels([])
        
        ax.set_box_aspect((self.nx, self.ny, np.max(scaled_prob_d)))
        ax.view_init(elev=40, azim=45)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(r'Probability Density, $|\psi_i(\mathbf{r})|^2$ (arb. units)', fontsize=label_fontsize, labelpad=20)
        cbar.ax.tick_params(labelsize=0)
        
        if save_path:
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        
        plt.show()
        
        return None
    
    # def H_eff_edge(self, steps_per_segment, vdT, N_theta, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, epsilonT=torch.pi, delta=None, initialise=False, fully_disorder=True):
        
    #     delete, eigenvalues_matrix, wf_matrix = self.quasienergies_states_edge(steps_per_segment, 'x', vdT, N_theta, rotation_angle=rotation_angle, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
    #     del delete
    #     log_eigenvalues = self.log_with_branchcut1(eigenvalues_matrix, epsilonT)
    #     log_eigenvalues = (1j / self.T) * log_eigenvalues

    #     # Initialize H_eff with the same shape and device as wf_matrix
    #     H_eff = torch.zeros_like(wf_matrix, dtype=torch.complex128)

    #     # Create diagonal matrices for all theta-points at once
    #     H_diag = torch.diag_embed(log_eigenvalues)

    #     # Ensure wf_matrix is complex
    #     wf_matrix = wf_matrix.to(torch.complex128)

    #     # Compute H_eff for all theta-points in one batch operation
    #     H_eff = torch.bmm(torch.bmm(wf_matrix, H_diag), wf_matrix.conj().transpose(-2, -1))

    #     return H_eff, log_eigenvalues, wf_matrix
    
    def deform_F_op(self, s, l1, l2):
        '''All y and l1 and l2 are counted from 1 instead of 0'''
        matrix_size = self.nx * self.ny
        diagonal = torch.zeros(matrix_size, device=self.device)
        
        for i in range(matrix_size):
            y = i // self.nx +1 # Get the y coordinate
            # print(y)
            if y <= l1:
                diagonal[i] = 0
            elif l1 < y <= l2:
                diagonal[i] = s * (y - l1) / (l2 - l1)
            elif l2 < y < (self.ny - l2):
                diagonal[i] = s
            elif (self.ny - l2) <= y <= (self.ny - l1):
                diagonal[i] = s * (self.ny - l1 - y) / (l2 - l1)
            else:  # y > (self.ny - l1)
                diagonal[i] = 0
        # print(diagonal)
        return torch.diag(diagonal)
    
    def compute_deformed_U_edge(self, l1, l2, t, theta_num, steps_per_segment, vdT, rotation_angle=torch.pi/4, epsilonT=torch.pi, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True, visualize=False, save_path=None):
        thetax, thetay = self.get_theta_values(theta_num, 'x')
        
        # First the time evolution operator defined on a cylinder
        U_t = self.time_evolution_operator1(t, steps_per_segment, 'x', vdT, rotation_angle, thetax, thetay, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        
        # Second, the H_eff that is defined on a torus
        H_eff, log_eigenvalues, S = self.H_eff(steps_per_segment, vdT, thetax, thetay, a, b, phi1_ex, phi2_ex, rotation_angle, epsilonT, delta, initialise, fully_disorder)
        del log_eigenvalues, S
        # Create the deformation operator F
        F_operator = self.deform_F_op(s=1, l1=l1, l2=l2).to(torch.complex128)
        # F_operator = F_operator @ F_operator
        # Compute the deformed effective Hamiltonian
        H_eff_deformed = F_operator @ H_eff @ F_operator
        
        # Compute exp(it H_eff_deformed)
        t = torch.as_tensor(t).to(H_eff_deformed.device)
        exp_H_eff_deformed = torch.matrix_exp(1j * t * H_eff_deformed)
        
        # Compute the final deformed evolution operator
        U_epsilon = U_t @ exp_H_eff_deformed

        # Check unitarity
        unitarity_error = torch.norm(U_epsilon @ U_epsilon.conj().transpose(-2, -1) - torch.eye(U_epsilon.shape[-1], device=U_epsilon.device, dtype=U_epsilon.dtype))
        print(f"Final unitarity error: {unitarity_error.item()}")

        if visualize:
            # Visualization function
            def visualize_matrix(matrix, save_path=None):
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.cpu().detach().numpy()

                if np.iscomplexobj(matrix):
                    matrix = np.abs(matrix)

                fig, ax = plt.subplots(figsize=(10, 8))
                
                im = ax.imshow(matrix, cmap='Purples', aspect='equal')

                # Adjust font sizes
                ax.tick_params(axis='both', which='major', labelsize=20)

                # Create a divider for existing axes instance
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                
                # Append axes to the right of the main axes, with 5% width of the main axes
                cax = divider.append_axes("right", size="5%", pad=0.05)

                # Create colorbar in the appended axes
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=20)
                cbar.set_label(r'$|U_{ij}|$', fontsize=26)
                if save_path:
                    plt.savefig(save_path, format='pdf', bbox_inches='tight')

                plt.show()
            # Visualize the deformed evolution operator
            visualize_matrix(U_epsilon[0], save_path)

        return U_epsilon
    
    # def separate_edge_operators(self, U_epsilon, threshold=1e-10):
    #     """
    #     Separate U_1,ε and U_2,ε from the full U_epsilon matrix by identifying the identity block in the middle.
        
    #     Parameters:
    #     U_epsilon (torch.Tensor): The full deformed evolution operator (complex128).
    #     threshold (float): Threshold to determine closeness to 1 for diagonal elements.
        
    #     Returns:
    #     tuple: (U_1_epsilon, U_2_epsilon)
    #     """
    #     N = U_epsilon.shape[-1]
        
    #     def is_identity(x):
    #         return torch.isclose(torch.abs(x), torch.tensor(1.0, dtype=torch.float64), atol=threshold)
        
    #     # Find the start of the identity block
    #     identity_start = 0
    #     while identity_start < N and not is_identity(U_epsilon[0, identity_start, identity_start]):
    #         identity_start += 1
        
    #     # Find the end of the identity block
    #     identity_end = N - 1
    #     while identity_end >= 0 and not is_identity(U_epsilon[0, identity_end, identity_end]):
    #         identity_end -= 1
        
    #     print(f"N: {N}")
    #     print(f"identity_start: {identity_start}")
    #     print(f"identity_end: {identity_end}")
        
    #     if identity_end < identity_start:
    #         print("Warning: identity_end is less than identity_start")
    #         print("Diagonal elements of U_epsilon:")
    #         print(torch.diag(U_epsilon[0]))
    #         return None, None  # or handle this case appropriately
        
    #     # Calculate the sizes of U_1 and U_2
    #     u1_size = identity_start
    #     u2_size = N - (identity_end + 1)
        
    #     # Balance the sizes
    #     edge_size = max(u1_size, u2_size)
        
    #     # Extract U_1,ε and U_2,ε with balanced sizes
    #     U_1_epsilon = U_epsilon[:, :edge_size, :edge_size]
    #     U_2_epsilon = U_epsilon[:, -edge_size:, -edge_size:]
        
    #     print(f"Shape of U_1_epsilon: {U_1_epsilon.shape}")
    #     print(f"Shape of U_2_epsilon: {U_2_epsilon.shape}")
        
    #     # Check off-diagonal elements in the supposed identity block
    #     if identity_end >= identity_start:
    #         off_diag_max = torch.max(torch.abs(U_epsilon[0, identity_start:identity_end+1, identity_start:identity_end+1] 
    #                                         - torch.eye(identity_end-identity_start+1, dtype=U_epsilon.dtype, device=U_epsilon.device)))
    #         print(f"Max off-diagonal element in identity block: {off_diag_max}")
    #     else:
    #         print("Cannot check off-diagonal elements: invalid identity block")
        
    #     return U_1_epsilon, U_2_epsilon
    
    # def separate_edge_operators(self, U_epsilon, l2, initial_threshold=1e-10, increment=1e-6):
    #     """
    #     Separate U_1,ε and U_2,ε from the full U_epsilon matrix by identifying the identity block in the middle.
        
    #     Parameters:
    #     U_epsilon (torch.Tensor): The full deformed evolution operator (complex128).
    #     l2 (int): Index offset for identity block determination.
    #     initial_threshold (float): Initial threshold to determine closeness to 1 for diagonal elements.
    #     increment (float): Increment to increase the threshold in case of failure.
        
    #     Returns:
    #     tuple: (U_1_epsilon, U_2_epsilon)
    #     """
    #     N = U_epsilon.shape[-1]

    #     def is_identity(x, threshold):
    #         return torch.isclose(torch.abs(x), torch.tensor(1.0, dtype=torch.float64), atol=threshold)

    #     threshold = initial_threshold
    #     while True:
    #         # Initialize identity_start and identity_end based on l2 and self.nx
    #         identity_start = l2 * self.nx
    #         # identity_start = 0 
    #         identity_end = N - 1 - self.nx * l2
    #         # identity_end = N - 1
    #         # Find the start of the identity block
    #         while identity_start < N and not is_identity(U_epsilon[0, identity_start, identity_start], threshold):
    #             identity_start += 1

    #         # Find the end of the identity block
    #         while identity_end >= 0 and not is_identity(U_epsilon[0, identity_end, identity_end], threshold):
    #             identity_end -= 1

    #         print(f"Threshold: {threshold}")
    #         print(f"N: {N}")
    #         print(f"identity_start: {identity_start}")
    #         print(f"identity_end: {identity_end}")

    #         if identity_end >= identity_start:
    #             break  # Exit the loop if the condition is satisfied

    #         # Increase the threshold
    #         threshold += increment

    #     if identity_end < identity_start:
    #         print("Failed to separate edge operators within the maximum number of attempts.")
    #         print("Diagonal elements of U_epsilon:")
    #         print(torch.diag(U_epsilon[0]))
    #         return None, None  # or handle this case appropriately

    #     # Calculate the sizes of U_1 and U_2
    #     u1_size = identity_start
    #     u2_size = N - (identity_end + 1)

    #     # Balance the sizes
    #     edge_size = max(u1_size, u2_size)

    #     # Extract U_1,ε and U_2,ε with balanced sizes
    #     U_1_epsilon = U_epsilon[:, :edge_size, :edge_size]
    #     U_2_epsilon = U_epsilon[:, -edge_size:, -edge_size:]

    #     print(f"Shape of U_1_epsilon: {U_1_epsilon.shape}")
    #     print(f"Shape of U_2_epsilon: {U_2_epsilon.shape}")

    #     # Check off-diagonal elements in the supposed identity block
    #     if identity_end >= identity_start:
    #         identity_block = U_epsilon[0, identity_start:identity_end+1, identity_start:identity_end+1]
    #         off_diag_max = torch.max(torch.abs(identity_block - torch.eye(identity_block.shape[0], dtype=U_epsilon.dtype, device=U_epsilon.device)))
    #         print(f"Max off-diagonal element in identity block: {off_diag_max}")
            
    #         # Detailed inspection of off-diagonal elements
    #         off_diag_elements = torch.abs(identity_block - torch.eye(identity_block.shape[0], dtype=U_epsilon.dtype, device=U_epsilon.device))
    #         print("Off-diagonal elements:")
    #         print(off_diag_elements)
    #     else:
    #         print("Cannot check off-diagonal elements: invalid identity block")

    #     return U_1_epsilon, U_2_epsilon
    
    def separate_edge_operators(self, U_epsilon):
        """
        Split a batch of matrices into upper and lower diagonal blocks.
        
        Parameters:
        U_epsilon (torch.Tensor): Batch of matrices to split. Shape: (batch_size, N, N)
        
        Returns:
        tuple: (U_1_epsilon, U_2_epsilon)
        """
        batch_size, N, _ = U_epsilon.shape
        
        # Calculate the midpoint
        mid = N // 2
        
        # Split the matrices
        U_1_epsilon = U_epsilon[:, :mid, :mid]
        U_2_epsilon = U_epsilon[:, mid:, mid:]
        
        print(f"Shape of U_1_epsilon: {U_1_epsilon.shape}")
        print(f"Shape of U_2_epsilon: {U_2_epsilon.shape}")
        
        return U_1_epsilon, U_2_epsilon
    
    def calculate_edge_winding_number(self, l1, l2, t, theta_num, steps_per_segment, vdT, 
                                  rotation_angle=torch.pi/4, epsilonT=torch.pi, 
                                  a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, 
                                  initialise=False, fully_disorder=True, visualize=False):
        """
        Calculate the edge winding numbers for U1 or U2 in the AFAI model.
        Returns:
        tuple: (n_edge_1, n_edge_2) Edge winding numbers for U1 (or U2)
        """
        # Compute the full deformed evolution operator
        U_epsilon = self.compute_deformed_U_edge(l1, l2, t, theta_num, steps_per_segment, vdT,
                                                rotation_angle, epsilonT, a, b, phi1_ex, phi2_ex,
                                                delta, initialise, fully_disorder, visualize)
        
        # Separate U1 and U2
        U1, U2 = self.separate_edge_operators(U_epsilon)

        def compute_winding_number(U_edge):
            # Compute step size
            dtheta = 2 * torch.pi / theta_num

            # Compute the derivative using central difference
            dU = (U_edge[2:] - U_edge[:-2]) / (2 * dtheta)

            # Compute U^dagger * dU/dtheta and take the trace in one operation
            integrand = torch.einsum('...ii', torch.matmul(U_edge[1:-1].conj().transpose(-2, -1), dU))
            # Print information about the integrand
            # print(f"Integrand shape: {integrand.shape}")
            # print(f"Integrand dtype: {integrand.dtype}")
            # print(f"First few values of integrand: {integrand[:5]}")
            # print(f"Mean of real part: {integrand.real.mean()}")
            # print(f"Mean of imaginary part: {integrand.imag.mean()}")
            # Sum over theta (which is equivalent to integration in the discrete case)
            n_edge = torch.sum(integrand.imag) * dtheta / (2 * torch.pi)

            return n_edge.item()

        n_edge_1 = compute_winding_number(U1)
        n_edge_2 = compute_winding_number(U2)

        return n_edge_1, n_edge_2
    
    def convergence_edge_invariant(self, l1, l2, t, grid_num_range, steps_per_segment, vdT, rotation_angle=torch.pi/4, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True, save_path=None):
        """
        Convergence test for the edge invariant with respect to the grid resolution in θ (theta_num).
        
        Parameters:
        - l1, l2: Indices used for the edge operator separation.
        - t: Time parameter.
        - grid_num_range: List or tensor of theta_num values to test for convergence.
        - steps_per_segment: Number of steps per segment (fixed across tests).
        - vdT: Fixed value of aperiodic strength to test.
        - rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder, tolerance: Parameters for the edge winding number calculation.
        - save_path: Path to save the plot (optional).
        
        Returns:
        - edge_invariant_values: Tensor of computed edge invariants for each grid size and branch cut value.
        """
        print('theta_num grid', grid_num_range)
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        edge_invariant_values = torch.zeros((len(grid_num_range), len(xi_values)), device=self.device)
        
        for i, theta_num in enumerate(grid_num_range):
            for j, xi in enumerate(xi_values):
                print(f"Computing for theta_num {theta_num}, ξ = {'0' if j == 0 else 'π'}")
                n_edge_1, n_edge_2 = self.calculate_edge_winding_number(
                    l1, l2, t, theta_num=theta_num, steps_per_segment=steps_per_segment, vdT=vdT,
                    rotation_angle=rotation_angle, epsilonT=xi,
                    a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex,
                    delta=delta, initialise=initialise, fully_disorder=fully_disorder,
                    visualize=False)
                n_edge_1 = torch.tensor(n_edge_1, device=self.device)
                n_edge_2 = torch.tensor(n_edge_2, device=self.device)
                
                # Get the larger value in absolute terms using PyTorch
                larger_value = torch.max(torch.abs(n_edge_1), torch.abs(n_edge_2))
                edge_invariant_values[i, j] = larger_value
                
                torch.cuda.empty_cache()
        
        # Convert to numpy arrays for plotting
        grid_np = grid_num_range.cpu().numpy() if isinstance(grid_num_range, torch.Tensor) else np.array(grid_num_range)
        edge_invariant_values_np = edge_invariant_values.cpu().numpy()
        
        # Plotting
        tick_label_fontsize = 32
        label_fontsize = 34
        legend_fontsize = 32
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(grid_np, edge_invariant_values_np[:, 0], label=r'$\varepsilon = 0$', marker='o')
        ax.plot(grid_np, edge_invariant_values_np[:, 1], label=r'$\varepsilon = \pi$', marker='s')
        ax.set_xlabel(r'$\theta_{\mathrm{num}}$', fontsize=label_fontsize)
        ax.set_ylabel(r'Edge Invariant $W_3[U_\varepsilon]$', fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.grid(True)
        
        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        
        plt.show()
        return edge_invariant_values
    
    def plot_edge_invariant_vs_vdT(self, l1, l2, t, theta_num, steps_per_segment, vdT_range, rotation_angle=torch.pi/4, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        """
        Generate a plot of edge invariant versus the aperiodic strength, vdT,
        for two branch cut values ξ=0 and π.
        """
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        edge_invariant_values = torch.zeros((len(vdT_range), len(xi_values)), device=self.device)

        for i, vdT in enumerate(vdT_range):
            for j, xi in enumerate(xi_values):
                n_edge_1, n_edge_2 = self.calculate_edge_winding_number(
                    l1, l2, t, theta_num, steps_per_segment, vdT,
                    rotation_angle=rotation_angle, epsilonT=xi,
                    a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex,
                    delta=delta, initialise=initialise, fully_disorder=fully_disorder,
                    visualize=False)
                n_edge_1 = torch.tensor(n_edge_1, device=self.device)
                n_edge_2 = torch.tensor(n_edge_2, device=self.device)

                # Get the larger value in absolute terms using PyTorch
                larger_value = torch.max(torch.abs(n_edge_1), torch.abs(n_edge_2))
                edge_invariant_values[i, j] = larger_value

            torch.cuda.empty_cache()

        # Convert to numpy arrays for plotting
        vdT_np = vdT_range.cpu().numpy() if isinstance(vdT_range, torch.Tensor) else np.array(vdT_range)
        edge_invariant_values_np = edge_invariant_values.cpu().numpy()
        tick_label_fontsize = 32
        label_fontsize = 34
        legend_fontsize = 32
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(vdT_np, edge_invariant_values_np[:, 0], label='ξ = 0', marker='o')
        plt.plot(vdT_np, edge_invariant_values_np[:, 1], label='ξ = π', marker='s')
        plt.xlabel('Aperiodic Strength (vdT)', fontsize=label_fontsize)
        plt.ylabel('Edge Invariant', fontsize=label_fontsize)
        # plt.title('Edge Invariant vs Aperiodic Strength')
        plt.ylim(-1.5, 1.5)
        plt.yticks(range(-1, 2))  # This sets integer ticks from -1 to 1
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_label_fontsize)
        plt.yticks(fontsize=tick_label_fontsize)
        plt.grid(True)
        plt.show()

        return edge_invariant_values

    def plot_edge_invariant_vs_vdT_disorder_realisation(self, l1, l2, t, theta_num, steps_per_segment, vdT_range, N_dis, rotation_angle=torch.pi/4, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True, tolerance=1e-10):
        """
        Generate a plot of disorder-averaged edge invariant versus the aperiodic strength, vdT,
        for two branch cut values ξ=0 and π.
        """
        xi_values = torch.tensor([0, torch.pi], device=self.device)
        edge_invariant_values = torch.zeros((len(vdT_range), len(xi_values)), device=self.device)

        for i, vdT in enumerate(vdT_range):
            for j, xi in enumerate(xi_values):
                edge_invariant_sum = 0
                for _ in range(N_dis):
                    self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
                    gc.collect()  # Force garbage collection
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

                    n_edge_1, n_edge_2 = self.calculate_edge_winding_number(
                        l1, l2, t, theta_num, steps_per_segment, vdT,
                        rotation_angle=rotation_angle, epsilonT=xi,
                        a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex,
                        delta=delta, initialise=initialise, fully_disorder=fully_disorder,
                        visualize=False, tolerance=tolerance
                    )
                    n_edge_1 = torch.tensor(n_edge_1, device=self.device)
                    n_edge_2 = torch.tensor(n_edge_2, device=self.device)

                    # Get the larger value in absolute terms using PyTorch
                    larger_value = torch.max(torch.abs(n_edge_1), torch.abs(n_edge_2))
                    edge_invariant_sum += larger_value

                    del n_edge_1, n_edge_2, larger_value
                    gc.collect()  # Force garbage collection
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                
                # Average over disorder realizations
                edge_invariant_values[i, j] = edge_invariant_sum / N_dis

            print(f"Completed vdT value {i+1}/{len(vdT_range)}, ξ = {'0' if j == 0 else 'π'}")  # Progress indicator

        # Convert to numpy arrays for plotting
        vdT_np = vdT_range.cpu().numpy() if isinstance(vdT_range, torch.Tensor) else np.array(vdT_range)
        edge_invariant_values_np = edge_invariant_values.cpu().numpy()
        
        # Plotting
        tick_label_fontsize = 32
        label_fontsize = 34
        legend_fontsize = 32

        plt.figure(figsize=(10, 6))
        plt.plot(vdT_np, edge_invariant_values_np[:, 0], label='ξ = 0', marker='o')
        plt.plot(vdT_np, edge_invariant_values_np[:, 1], label='ξ = π', marker='s')
        plt.xlabel('Aperiodic Strength (vdT)', fontsize=label_fontsize)
        plt.ylabel('Edge Invariant', fontsize=label_fontsize)
        plt.ylim(-1.5, 1.5)
        plt.yticks(range(-1, 2))
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_label_fontsize)
        plt.yticks(fontsize=tick_label_fontsize)
        plt.grid(True)
        plt.show()
        return edge_invariant_values
    
    ## Evolving a delta wavefunction --> Transmission probability
    def infinitesimal_evol_operator(self, Hr, V_dis, dt):
        '''The infinitesimal evolution operator for the real space Hamiltonian'''
        U = torch.matrix_exp(-1j * (Hr) * dt/2) @ torch.matrix_exp(-1j * (V_dis) * dt) @ torch.matrix_exp(-1j * (Hr) * dt/2)
        return U
    
    def real_time_trans(self, N_times, steps_per_segment, initial_position, tbc, vdT, rotation_angle=torch.pi/4, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        """The real time transmission Amplitude of evolved initial wavepacket at given initial_position (input) over N_times period"""
        U = self.time_evolution_operator1(self.T * N_times, steps_per_segment, tbc, vdT, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)
        
        batch_size = U.shape[0]
        num_sites = U.shape[1]
        
        # Create initial state vector for each batch
        vector = torch.zeros((batch_size, num_sites), dtype=torch.complex128, device=self.device)
        vector[:, initial_position - 1] = 1.0
        
        # Perform batch matrix-vector multiplication
        Ua = torch.bmm(U, vector.unsqueeze(-1)).squeeze(-1)
        
        return Ua
    
    def transmission_prob(self, N_max, steps_per_segment, initial_position, energy, tbc, vdT, rotation_angle=torch.pi/4, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        # Ensure vdT is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        elif isinstance(vdT, torch.Tensor):
            vdT = vdT.to(self.device)

        # Ensure energy is a tensor
        if isinstance(energy, (int, float)):
            energy = torch.tensor([energy], device=self.device)
        elif not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=self.device)
        elif isinstance(energy, torch.Tensor):
            energy = energy.to(self.device)

        # Multiply energy by 2π/T
        energy = energy * (2 * torch.pi / self.T)

        vdT_size = vdT.shape[0]
        num_sites = self.nx * self.ny

        # Reshape vdT, energy for broadcasting
        vdT = vdT.view(vdT_size, 1)

        # Compute U for one period
        U_one_period = self.time_evolution_operator1(self.T, steps_per_segment, tbc, vdT, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)

        # Compute powers of U_one_period for all required periods
        U_powers = [torch.eye(num_sites, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(vdT_size, -1, -1)]
        for _ in range(N_max):
            U_powers.append(torch.matmul(U_powers[-1], U_one_period))
        # Stack all U_powers
        U_all_periods = torch.stack(U_powers)
        # Create initial state vector
        vector = torch.zeros((vdT_size, num_sites), dtype=torch.complex128, device=self.device)
        vector[:, initial_position - 1] = 1.0

        # Compute G for all periods at once
        G_all = torch.matmul(U_all_periods, vector.unsqueeze(-1)).squeeze(-1)
        G_all = G_all.view(N_max + 1, vdT_size, num_sites)  # Shape [periods, vdT, number_sites]

        # Compute complex exponentials for all periods at once
        nn = torch.arange(N_max + 1, device=self.device).view(1, -1, 1)  # Shape [energy, Period, vdT, 1, 1]
        energy = energy.view(-1, 1, 1)
        complex_exponent = 1j * energy * nn * self.T
        # Compute G_aa
        summm = G_all.unsqueeze(0) * torch.exp(complex_exponent).unsqueeze(-1)
        G_aa = torch.sum(summm, dim=1) / (N_max + 1)

        transmission_prob = torch.abs(G_aa)**2
        transmission_prob = transmission_prob.permute(1, 0, 2)

        return transmission_prob  # shape: (vdT_size, energy_size, num_sites)
    
    def transmission_prob_batched(self, N_max, steps_per_segment, initial_position, energy, tbc, vdT, max_batch_size=10, rotation_angle=torch.pi/4, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        # Ensure vdT is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        elif isinstance(vdT, torch.Tensor):
            vdT = vdT.to(self.device)

        # Ensure energy is a tensor
        if isinstance(energy, (int, float)):
            energy = torch.tensor([energy], device=self.device)
        elif not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=self.device)
        elif isinstance(energy, torch.Tensor):
            energy = energy.to(self.device)

        # Multiply energy by 2π/T
        energy = energy * (2 * torch.pi / self.T)

        vdT_size = vdT.shape[0]
        num_sites = self.nx * self.ny

        # Reshape vdT, energy for broadcasting
        vdT = vdT.view(vdT_size, 1)
        energy = energy.view(-1, 1, 1)

        # Compute U for one period
        U_one_period = self.time_evolution_operator1(self.T, steps_per_segment, tbc, vdT, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)

        # Create initial state vector
        vector = torch.zeros((vdT_size, num_sites), dtype=torch.complex128, device=self.device)
        vector[:, initial_position - 1] = 1.0

        # Initialize result accumulator
        G_sum = torch.zeros((energy.shape[0], vdT_size, num_sites), dtype=torch.complex128, device=self.device)

        # Initialize U_power as identity matrix
        U_power = torch.eye(num_sites, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(vdT_size, -1, -1)

        # Process in batches
        for start in range(0, N_max + 1, max_batch_size):
            end = min(start + max_batch_size, N_max + 1)
            batch_size = end - start

            # Compute powers of U for this batch
            U_powers = []
            for _ in range(batch_size):
                U_powers.append(U_power)
                U_power = torch.matmul(U_power, U_one_period)
            U_powers = torch.stack(U_powers)
            # Compute G for this batch
            G_batch = torch.matmul(U_powers, vector.unsqueeze(-1)).squeeze(-1)
            
            # Compute complex exponentials for this batch
            nn = torch.arange(start, end, device=self.device).view(1, -1, 1)
            complex_exponent = 1j * energy * nn * self.T
            exp = torch.exp(complex_exponent)
            # Compute and accumulate to G_sum
            G_sum += torch.sum(G_batch.unsqueeze(0) * exp.unsqueeze(-1), dim=1)

        # Compute final G_aa
        G_aa = G_sum / (N_max + 1)

        transmission_prob = torch.abs(G_aa)**2
        transmission_prob = transmission_prob.permute(1, 0, 2)

        return transmission_prob  # shape: (vdT_size, energy_size, num_sites)
    
    ## The following two functions "real_trans_prob_avg_dis" and "trans_prob_avg_disorder_realisation" only deal with the fully disordered case
    def real_trans_prob_avg_dis(self, N_dis, N_times, steps_per_segment, initial_pos, tbc, vdT, delta=None):
        '''The disorder averaged real-time transmission probability of the evolved initial wavepacket at given initial_position (input)
        over N_times period averaging over N_dis realisation'''
        '''The first initial_position is 1 instead of 0'''
        
        # Ensure vdT is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        
        batch_size = vdT.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((batch_size, num_sites), device=self.device)
        
        for N in range(N_dis):
            self.H_disorder_cached = None
            print(N)
            
            real_amp = self.real_time_trans(N_times, steps_per_segment, initial_pos, tbc, vdT,  delta=delta, fully_disorder=True)
            
            avg += torch.abs(real_amp)**2
            
            del real_amp
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_dis
        
        del avg
        torch.cuda.empty_cache()
        
        return avg_tp   # shape: a 3D tensor with dimensions (vdT_size, energy_size, total number of sites)
    
    def trans_prob_avg_disorder_realisation(self, N_dis, N_max, steps_per_segment, initial_position, energy, tbc, vdT, delta=None):
        """The disorder averaged transmission probability of the evolved initial wavepacket at given initial_position (input) over N_max period averaging over N_dis realisation"""
        '''The first initial_position is 1 instead of 0'''
        '''The output should be a tensor with dimensions (vdT_size, energy_size, total numbers of sites)'''
        
        # Ensure vdT is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        
        # Ensure energy is a tensor
        if isinstance(energy, (int, float)):
            energy = torch.tensor([energy], device=self.device)
        elif not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=self.device)
        
        vdT_size = vdT.shape[0]
        energy_size = energy.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((vdT_size, energy_size, num_sites), device=self.device)
        
        for N in range(N_dis):
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            print('number of disorder realisation', N)
            
            trans_prob_single = self.transmission_prob(N_max, steps_per_segment, initial_position, energy, tbc, vdT, delta=delta, fully_disorder=True)
            # Explicitly squeeze out the singleton dimension
            # trans_prob_single = trans_prob_single.squeeze(2)
            
            avg += trans_prob_single  # Sum up the averages
            
            del trans_prob_single  # Free up memory of the single average once added
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_dis
        
        del avg
        torch.cuda.empty_cache()
        
        return avg_tp.squeeze()
    
    ## The following two functions "real_trans_prob_avg_phase" and "trans_prob_avg_phase_realisation" only deal with the aperiodic case
    def real_trans_prob_avg_phase(self, N_phi, N_times, steps_per_segment, initial_pos, tbc, vdT, rotation_angle=np.pi/4, theta_x=0, theta_y=0, a=0, b=0, delta=None):
        # Ensure vdT is a tensor
        if isinstance(vdT, (int, float)):
            vdT = torch.tensor([vdT], device=self.device)
        elif not isinstance(vdT, torch.Tensor):
            vdT = torch.tensor(vdT, device=self.device)
        
        batch_size = vdT.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((batch_size, num_sites), device=self.device)
        phi1_vals = np.random.uniform(0, 2*np.pi, N_phi)
        phi2_vals = np.random.uniform(0, 2*np.pi, N_phi)
        
        for N in range(N_phi):
            print(N)
            real_amp = self.real_time_trans(N_times, steps_per_segment, initial_pos, tbc, vdT, rotation_angle, 
                                            theta_x, theta_y, a, b, phi1=phi1_vals[N], phi2=phi2_vals[N], 
                                            delta=delta, fully_disorder=False)
            avg += torch.abs(real_amp)**2
            del real_amp
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_phi
        del avg
        torch.cuda.empty_cache()
        return avg_tp

    def trans_prob_avg_phase_realisation(self, N_phi, N_max, steps_per_segment, initial_pos, energy, tbc, vdT, rotation_angle=np.pi/4, theta_x=0, theta_y=0, a=0, b=0, delta=None):
        # Ensure vdT and energy are tensors
        vdT = torch.tensor(vdT, device=self.device) if not isinstance(vdT, torch.Tensor) else vdT
        energy = torch.tensor(energy, device=self.device) if not isinstance(energy, torch.Tensor) else energy
        
        vdT_size, energy_size = vdT.shape[0], energy.shape[0]
        num_sites = self.nx * self.ny
        avg = torch.zeros((vdT_size, energy_size, num_sites), device=self.device)
        # Generate all phi1 and phi2 values at once
        phi1_vals = torch.from_numpy(np.random.uniform(0, 2*np.pi, N_phi)).to(self.device)
        phi2_vals = torch.from_numpy(np.random.uniform(0, 2*np.pi, N_phi)).to(self.device)

        # Compute transmission probabilities
        for N in range(N_phi):
            print('number of phase realisation', N)
            trans_prob = self.transmission_prob(N_max, steps_per_segment, initial_pos, energy, tbc, vdT, rotation_angle, 
                                                    theta_x, theta_y, a, b, phi1=phi1_vals[N], phi2=phi2_vals[N], 
                                                    delta=delta, fully_disorder=False)
            avg += trans_prob
            del trans_prob
            torch.cuda.empty_cache()
            
        avg_tp = avg/ N_phi
        del avg
        torch.cuda.empty_cache()
        
        return avg_tp
    
    def plot_transmission_probability(self, transmission_prob, figsize, save_path=None):
        """Plot the transmission probability of the intial wavepacket at given initial position (input)"""
        '''The input should be a vector with dimension equal to the total number of sites'''
        fig, ax = plt.subplots(figsize=figsize)
        tick_label_fontsize = 32
        label_fontsize = 40
        trans_prob_cpu = transmission_prob.cpu().numpy().reshape(self.ny, self.nx)
        norm = plt.Normalize(np.min(trans_prob_cpu), np.max(trans_prob_cpu))
        cmap = plt.get_cmap('viridis')
        plt.imshow(trans_prob_cpu, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.xlabel('X', fontsize=label_fontsize)
        plt.ylabel('Y', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)
        plt.gca().invert_yaxis()
        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        return None

    def plot_transmission_probability1(self, transmission_prob, figsize, quasi=False, scale_factor=5e4, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, grid_density=100, fix_z_scale=True, save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        tick_label_fontsize = 32
        label_fontsize = 34
        
        trans_prob_cpu = transmission_prob.cpu().numpy().reshape(self.ny, self.nx)
        scaled_trans_prob = trans_prob_cpu * scale_factor

        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y)
        
        ls = LightSource(azdeg=315, altdeg=45)
        
        # Use the custom colormap
        cmap = LinearSegmentedColormap.from_list('white_purple', ['white', 'purple'], N=100)
        
        surf = ax.plot_surface(X, Y, scaled_trans_prob, 
                            cmap=cmap,
                            rstride=1, cstride=1,
                            linewidth=0,
                            antialiased=False,
                            shade=True, lightsource=ls)
        
        if quasi:
            # Generate quasiperiodic potential
            rotation_angle = rotation_angle.to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)
            x_dense = np.linspace(-1, self.nx, grid_density) - (self.nx - 1) / 2
            y_dense = np.linspace(-1, self.ny, grid_density) - (self.ny - 1) / 2
            X_dense, Y_dense = np.meshgrid(x_dense + (self.nx - 1) / 2, y_dense + (self.ny - 1) / 2, indexing='ij')
            Z = self.quasip_continuum(x_dense, y_dense, a, b, phi1_ex, phi2_ex, rotation_angle)

            # Add quasiperiodic contour plot at the bottom
            offset = -np.max(scaled_trans_prob) * 1.1  # Set the offset to 10% below the minimum of the transmission probability
            quasi_contour = ax.contourf(X_dense, Y_dense, Z, levels=100, zdir='z', offset=offset, cmap='viridis', alpha=0.7)
            
            # Add colorbar for quasiperiodic potential
            cbar_quasi = fig.colorbar(quasi_contour, ax=ax, shrink=0.5, aspect=5, pad=0.15)
            cbar_quasi.set_label('Quasiperiodic Potential', fontsize=label_fontsize, labelpad=20)
            cbar_quasi.ax.tick_params(labelsize=tick_label_fontsize)

        if fix_z_scale:
            if quasi:
                ax.set_zlim((offset, np.max(scaled_trans_prob)))
            else:
                ax.set_zlim((0, np.max(scaled_trans_prob)))
        
        ax.set_xlabel('X', labelpad=200, fontsize=label_fontsize)
        ax.set_ylabel('Y', labelpad=50, fontsize=label_fontsize)
        
        # Set 5 equally spaced tick labels for x and y axes
        x_ticks = np.linspace(0, self.nx-1, 5, dtype=int)
        y_ticks = np.linspace(0, self.ny-1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        
        # Remove z-axis tick labels
        ax.set_zticklabels([])
        
        ax.set_box_aspect((self.nx, self.ny, np.max(scaled_trans_prob)))
        ax.view_init(elev=30, azim=45)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(r'$\langle |G_N(r,r_0,\epsilon)|^2 \rangle$ (arb. units)', 
               fontsize=label_fontsize, labelpad=20)
        cbar.ax.tick_params(labelsize=0)
        
        if save_path:
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        
        plt.show()
        
        return None
    
    ## Quantised Charge pumping
    def derivative_H_tbc1(self, theta_y, tbc='y'):
        if isinstance(theta_y, (int, float)):
            theta_y = torch.tensor([theta_y], device=self.device)
        elif not isinstance(theta_y, torch.Tensor):
            theta_y = torch.tensor(theta_y, device=self.device)
        else:
            theta_y = theta_y.to(self.device)
        is_batch = theta_y.dim() > 1 or (theta_y.dim() == 1 and theta_y.shape[0] > 1)
        batch_size = theta_y.shape[0] if is_batch else 1
        theta_y = theta_y.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H1 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        if tbc == 'y' or tbc == 'xy':
            p = 0
            d_phase = 1j * torch.exp(1j * theta_y).squeeze()
            while 1 + 2 * p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[:, int(a), int(b)] = -J_coe_tensor * d_phase
                H1[:, int(b), int(a)] = -J_coe_tensor * d_phase.conj()
                p += 1
        return H1.squeeze() if not is_batch else H1
    
    def derivative_H_tbc2(self, theta_x, tbc='x'):
        if isinstance(theta_x, (int, float)):
            theta_x = torch.tensor([theta_x], device=self.device)
        elif not isinstance(theta_x, torch.Tensor):
            theta_x = torch.tensor(theta_x, device=self.device)
        else:
            theta_x = theta_x.to(self.device)
        is_batch = theta_x.dim() > 1 or (theta_x.dim() == 1 and theta_x.shape[0] > 1)
        batch_size = theta_x.shape[0] if is_batch else 1
        theta_x = theta_x.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H2 = torch.zeros((batch_size , size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        if tbc == 'x' or tbc == 'xy':
            p = 0
            phase = 1j * torch.exp(1j * theta_x).squeeze()
            while self.nx - 1 + 2 * self.nx * p < size and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                H2[:, a, b] = -J_coe_tensor * phase
                H2[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        return H2.squeeze() if not is_batch else H2
    
    def derivative_H_tbc3(self, theta_y, tbc='y'):
        if isinstance(theta_y, (int, float)):
            theta_y = torch.tensor([theta_y], device=self.device)
        elif not isinstance(theta_y, torch.Tensor):
            theta_y = torch.tensor(theta_y, device=self.device)
        else:
            theta_y = theta_y.to(self.device)
        is_batch = theta_y.dim() > 1 or (theta_y.dim() == 1 and theta_y.shape[0] > 1)
        batch_size = theta_y.shape[0] if is_batch else 1
        theta_y = theta_y.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H3 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 2 * p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                phase = 1j * torch.exp(1j * theta_y).squeeze()
                H3[:, int(a), int(b)] = -J_coe_tensor * phase
                H3[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        return H3.squeeze() if not is_batch else H3
    
    def derivative_H_tbc4(self, theta_x, tbc='x'):
        if isinstance(theta_x, (int, float)):
            theta_x = torch.tensor([theta_x], device=self.device)
        elif not isinstance(theta_x, torch.Tensor):
            theta_x = torch.tensor(theta_x, device=self.device)
        else:
            theta_x = theta_x.to(self.device)
        is_batch = theta_x.dim() > 1 or (theta_x.dim() == 1 and theta_x.shape[0] > 1)
        batch_size = theta_x.shape[0] if is_batch else 1
        theta_x = theta_x.view(batch_size, 1, 1)
        size = self.nx * self.ny
        H4 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while 2 * self.nx * (1 + p) - 1 < size and self.nx % 2 == 0:
                a = 2 * self.nx * (1 + p) - 1
                b = 2 * self.nx * p + self.nx
                phase = 1j * torch.exp(1j * theta_x).squeeze()
                H4[:, a, b] = -J_coe_tensor * phase
                H4[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        return H4.squeeze() if not is_batch else H4
    
    def derivative_H_tbc(self, t_batch, tbc, theta_x, theta_y):
        """
        The derivative of Hamiltonian H(t) wrt theta with twisted boundary conditions in either x, y
        for a batch of time values or a single time value.
        
        Parameters:
        t_batch (torch.Tensor or float): A tensor of time values or a single time value.
        tbc (torch.Tensor): Twisted boundary conditions.
        theta_x (torch.Tensor): Theta_x values.
        theta_y (torch.Tensor): Theta_y values.

        Returns:
        torch.Tensor: Derivative of the Hamiltonian with respect to the twisted boundary conditions.
        """
        # Convert inputs to tensors and reshape
        t_batch = torch.as_tensor(t_batch, device=self.device).reshape(-1)
        theta_x = torch.as_tensor(theta_x, device=self.device).reshape(-1, 1, 1, 1)
        theta_y = torch.as_tensor(theta_y, device=self.device).reshape(1, -1, 1, 1)
        
        # Ensure t is within [0, T)
        t_batch = t_batch % self.T

        size = self.nx * self.ny
        n_times = t_batch.shape[0]
        n_theta_x = theta_x.shape[0]
        n_theta_y = theta_y.shape[1]

        # Pre-allocate the output tensor
        H = torch.zeros(n_theta_x, n_theta_y, n_times, size, size, dtype=torch.cdouble, device=self.device)
        # print("shape of H", H.shape)
        # Compute H_tbc for each time step
        for idx, t in enumerate(t_batch):
            if t < self.T / 5:
                H_tbc = self.derivative_H_tbc1(theta_y.squeeze(), tbc)
                # print("Before expand H_tbc1:", H_tbc.shape)
                H_tbc_expanded = H_tbc.unsqueeze(0).unsqueeze(0).expand(n_theta_x, n_theta_y, size, size)
                # print("After expand H_tbc1:", H_tbc_expanded.shape)
                H[:, :, idx] = H_tbc_expanded
            elif self.T / 5 <= t < 2 * self.T / 5:
                H_tbc = self.derivative_H_tbc2(theta_x.squeeze(), tbc)
                # print("Before expand H_tbc2:", H_tbc.shape)
                H_tbc_expanded = H_tbc.unsqueeze(0).unsqueeze(1).expand(n_theta_x, n_theta_y, size, size)
                # print("After expand H_tbc2:", H_tbc_expanded.shape)
                H[:, :, idx] = H_tbc_expanded
            elif 2 * self.T / 5 <= t < 3 * self.T / 5:
                H_tbc = self.derivative_H_tbc3(theta_y.squeeze(), tbc)
                # print("Before expand H_tbc3:", H_tbc.shape)
                H_tbc_expanded = H_tbc.unsqueeze(0).unsqueeze(0).expand(n_theta_x, n_theta_y, size, size)
                # print("After expand H_tbc3:", H_tbc_expanded.shape)
                H[:, :, idx] = H_tbc_expanded
            elif 3 * self.T / 5 <= t < 4 * self.T / 5:
                H_tbc = self.derivative_H_tbc4(theta_x.squeeze(), tbc)
                # print("Before expand H_tbc4:", H_tbc.shape)
                H_tbc_expanded = H_tbc.unsqueeze(0).unsqueeze(1).expand(n_theta_x, n_theta_y, size, size)
                # print("After expand H_tbc4:", H_tbc_expanded.shape)
                H[:, :, idx] = H_tbc_expanded
            else:  # 4 * self.T / 5 <= t < self.T
                H[:, :, idx] = torch.zeros(size, size, dtype=torch.cdouble, device=self.device)

        return H.squeeze() ## The most general case has the shape (theta_x, t, size, size)
    
    def floquet_state_t(self, vdT_tensor, theta_x, N_div, steps_per_segment, n=1, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        '''
        Computes Floquet states, adapting to input dimensions (with/without vdT and/or theta_x).
        '''
        # Get the size of the system
        size = self.nx * self.ny
        # Ensure vdT_tensor and theta_x are tensors
        if not isinstance(vdT_tensor, torch.Tensor):
            vdT_tensor = torch.tensor(vdT_tensor, device=self.device)
        if not isinstance(theta_x, torch.Tensor):
            theta_x = torch.tensor(theta_x, device=self.device)
        
        # Add a dimension if theta_x is a scalar
        if theta_x.dim() == 0:
            theta_x = theta_x.unsqueeze(0)
        if vdT_tensor.dim() == 0:
            vdT_tensor = vdT_tensor.unsqueeze(0)
        # Create a time tensor
        t = torch.linspace(0, n * self.T, N_div + 1, device=self.device)

        # Compute the time evolution operator
        U_tensor = self.time_evolution_operator1(self.T, steps_per_segment, 'x', vdT_tensor, rotation_angle, theta_x, theta_y=0, a=a, b=b, phi1=phi1_ex, phi2=phi2_ex, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
        # Compute the time evolution operator that is depend on t
        U_t = self.time_evolution_operator1(t, steps_per_segment, 'x', vdT_tensor, rotation_angle, theta_x, theta_y=0, a=a, b=b, phi1=phi1_ex, phi2=phi2_ex, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
        U_t = U_t.view(vdT_tensor.shape[0], theta_x.shape[0], t.shape[0], size, size)
        # print(U_t.shape)
        # Perform eigen decomposition
        eigvals, eigvecs = torch.linalg.eig(U_tensor)
        # print("Original eigvecs shape:", eigvecs.shape)
        # print("Original eigvecs norm:", torch.norm(eigvecs, dim=-2))
        # print("Original eigvecs (first point):", eigvecs)
        # Compute and sort eigenvalues
        eigv_r = (-1j * torch.log(eigvals)).real
        # Reshape tensors to the most general case (vdT, theta_x, size)
        eigv_r = eigv_r.view(vdT_tensor.shape[0], theta_x.shape[0], size)
        eigvecs = eigvecs.view(vdT_tensor.shape[0], theta_x.shape[0], size, size)
        sorted_indices = torch.argsort(eigv_r, dim=-1)
        # Apply the sorting to eigvecs
        expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
        eigvecs_sorted = torch.gather(eigvecs, -1, expanded_indices)
        # If need to squeeze the results
        # eigvecs_sorted = eigvecs_sorted.squeeze()
        # print("Sorted eigvecs shape:", eigvecs_sorted.shape)
        # print("Sorted eigvecs norm:", torch.norm(eigvecs_sorted, dim=-2))
        # print("Sorted eigvecs (first point):", eigvecs_sorted)
        # Check orthogonality
        # dot_products = torch.matmul(eigvecs_sorted.transpose(-1, -2).conj(), eigvecs_sorted)
        # off_diagonal = dot_products - torch.eye(dot_products.shape[-1], device=dot_products.device)
        # print("Max off-diagonal element:", torch.max(torch.abs(off_diagonal)))
        # Reshape to the general case for further processing
        eigvecs_sorted = eigvecs_sorted.view(vdT_tensor.shape[0], theta_x.shape[0], 1, size, size)
        # print(E_T_sorted.shape)
        # print(eigvecs_sorted.shape)
        # Apply phase to get Floquet states
        floquet_states = U_t @ eigvecs_sorted
        # print(floquet_states.shape)
        # Squeeze out any singleton dimensions
        floquet_states = floquet_states.squeeze()
        # print(floquet_states.shape)
        return t, floquet_states # The most general case of the sorted_eigvecs has shape: (vdT, theta_x, t, size, size)

    def integrand(self, floquet_states_t, derivative_H):
        """
        Calculate the <psi(t) | derivative_H | psi(t)> for each time t and potentially each theta_x.

        Parameters:
        floquet_states_t (torch.Tensor): Tensor of shape (t, size, size) or (theta_x, t, size, size) 
                                        containing Floquet states at different times and potentially different theta_x.
        derivative_H (torch.Tensor): Tensor of shape (t, size, size) or (theta_x, t, size, size) 
                                    containing the derivative of the Hamiltonian with respect to theta_x 
                                    at different times and potentially different theta_x.

        Returns:
        torch.Tensor: Tensor of shape (t, size) or (theta_x, t, size) containing the expectation values 
                    for each time step and potentially each theta_x.
        """
        # Ensure the inputs are on the same device
        device = floquet_states_t.device

        # Check if we have a theta_x dimension
        has_theta_x = floquet_states_t.dim() == 4

        # Compute the Hermitian conjugate (dagger) of the Floquet states
        floquet_states_t_dagger = floquet_states_t.conj().transpose(-2, -1)

        # Compute the expectation value using einsum
        if has_theta_x:
            # Shape: (theta_x, t, size, size)
            expectation_values = torch.einsum('xtij,xtjk,xtkl->xtil', 
                                            floquet_states_t_dagger, derivative_H, floquet_states_t)
        else:
            # Shape: (t, size, size)
            expectation_values = torch.einsum('tij,tjk,tkl->til', 
                                            floquet_states_t_dagger, derivative_H, floquet_states_t)
        # For floating-point tensors, consider a small epsilon for comparison

        # The diagonal elements give <psi(t) | derivative_H | psi(t)>
        expectation_values_diagonal = torch.diagonal(expectation_values, dim1=-2, dim2=-1)

        return expectation_values_diagonal.real

    def simpson_integration(self, vdT_tensor, theta_x, N_div, steps_per_segment, n=1, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        """
        Perform Simpson's rule integration of the expectation value <psi(t)|derivative_H|psi(t)> over time,
        including the calculation of Floquet states and derivative of Hamiltonian.

        Parameters:
        vdT_tensor (torch.Tensor): Tensor for vdT values
        theta_x (torch.Tensor): Theta_x values
        N_div (int): Number of divisions for time (must be even to ensure odd number of points)
        steps_per_segment (int): Steps per segment for time evolution
        n (int): Number of periods to integrate over
        a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder: 
            Additional parameters for floquet_state_t and derivative_H_tbc

        Returns:
        torch.Tensor: Integrated expectation value over time. Shape will be (theta_x, size) or (size)
        """
        # Ensure odd number of points for Simpson's rule
        if N_div % 2 != 0:
            raise ValueError("N_div must be even to ensure an odd number of time points for Simpson's rule.")

        # Calculate Floquet states
        t, floquet_states = self.floquet_state_t(vdT_tensor, theta_x, N_div, steps_per_segment, n, a, b, 
                                                phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)

        # Calculate derivative of Hamiltonian
        derivative_H = self.derivative_H_tbc(t, 'x', theta_x, theta_y=0)

        # Calculate the integrand
        y = self.integrand(floquet_states, derivative_H)
        # print(y)
        # Get the step size
        dt = t[1] - t[0]

        # Apply Simpson's rule
        if y.dim() == 2:  # Shape: (t, size)
            s = y[0] + y[-1] + 4 * torch.sum(y[1:-1:2], dim=0) + 2 * torch.sum(y[2:-2:2], dim=0)
        else:  # Shape: (theta_x, t, size)
            s = y[:, 0] + y[:, -1] + 4 * torch.sum(y[:, 1:-1:2], dim=1) + 2 * torch.sum(y[:, 2:-2:2], dim=1)

        result = dt / 3 * s

        return result, floquet_states
    
    def single_p_wf_ini(self):
        """
        Initialize a single-particle wavefunction that fills the first half of the sites.
        
        Returns:
        torch.Tensor: Initialized single-particle wavefunction.
        """
        size = self.nx * self.ny
        if size % 2 != 0:
            raise ValueError("Size must be an even number.")
        # Create a wavefunction array with ones for the first half and zeros for the second half
        half_size = size // 2
        # Create a wavefunction array with normalized values for the first half and zeros for the second half
        wavefunction = torch.zeros(size, dtype=torch.cdouble, device=self.device)
        wavefunction[:half_size] = 1 / torch.sqrt(torch.tensor(half_size, dtype=torch.cdouble, device=self.device))
        # wavefunction[:half_size] = 1 
        return wavefunction
    
    def single_wf_dic(self, size, l):
        """
        Create a matrix with size (row) * l (column) to store the single-particle wavefunctions.

        Parameters:
        size (int): Total number of sites.
        l (int): Number of occupied sites (columns to keep).

        Returns:
        torch.Tensor: Matrix with the first l columns of the identity matrix.
        """
        # Create the identity matrix
        idn = torch.eye(size)
        
        # Slice to keep the first l columns
        dic = idn[:, :l]
        
        return dic
    
    def levi_civita(self, permutation):
        """
        Calculate the Levi-Civita symbol for a given permutation.

        Parameters:
        permutation (tuple): A permutation of indices.

        Returns:
        int: The Levi-Civita symbol (1 or -1).
        """
        n = len(permutation)
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[i] > permutation[j]:
                    sign *= -1
        return sign
    
    def construct_slater_determinant(self, l):
        """
        Construct the Slater determinant for a system with a given number of sites and occupied states.

        Parameters:
        l (int): Number of occupied sites (fermions).

        Returns:
        torch.Tensor: Slater determinant wavefunction.
        """
        size = self.nx * self.ny
        # Get the single-particle wavefunctions matrix
        wf_matrix = self.single_wf_dic(size, l)
        
        # Initialize the Slater determinant wavefunction
        slater_det = torch.zeros((size**l,))
        
        # Generate all permutations of the column indices
        indices = list(range(l))
        all_permutations = permutations(indices)
        
        # Compute the Slater determinant
        for perm in all_permutations:
            sign = self.levi_civita(perm)
            if sign != 0:
                term = wf_matrix[:, perm[0]]
                for idx in perm[1:]:
                    term = torch.kron(term, wf_matrix[:, idx])
                slater_det += sign * term
        # Normalize the Slater determinant using the known factor 1/sqrt(l!)
        normalization_factor = 1 / math.sqrt(math.factorial(l))
        slater_det *= normalization_factor

        return slater_det
    
    def slater_determinant(self, l):
        """
        Construct the Slater determinant for a system with a given number of sites and occupied states.

        Parameters:
        size (int): Total number of sites.
        l (int): Number of occupied sites (fermions).

        Returns:
        torch.Tensor: Slater determinant wavefunction.
        """
        size = self.nx * self.ny
        # Get the single-particle wavefunctions matrix
        wf_matrix = self.single_wf_dic(size, l)
        
        # Initialize the Slater determinant wavefunction list
        slater_det_list = []
        
        # Generate all permutations of the column indices
        indices = list(range(l))
        all_permutations = permutations(indices)
        
        # Compute the Slater determinant
        for perm in all_permutations:
            sign = self.levi_civita(perm)
            term = wf_matrix[:, perm[0]]
            for idx in perm[1:]:
                term = torch.kron(term, wf_matrix[:, idx])
            slater_det_list.append(sign * term)
            del term  # Free memory for the intermediate tensor
            gc.collect()  # Ensure garbage collection
        
        # Sum the terms to form the final Slater determinant wavefunction
        slater_det = sum(slater_det_list)
        del slater_det_list  # Free memory for the list
        gc.collect()  # Ensure garbage collection
        
        # Normalize the Slater determinant using the known factor 1/sqrt(l!)
        normalization_factor = 1 / math.sqrt(math.factorial(l))
        slater_det *= normalization_factor
        
        return slater_det

    def nj_single_particle(self, initial_state, floquet_states_t):
        """
        Calculate nj for each single-particle Floquet state at t=0.
        
        Parameters:
        initial_state (torch.Tensor): The initial single-particle state (1D tensor)
        floquet_states_t (torch.Tensor): 3D or 4D tensor of Floquet states
        Shape: (t, size, size) or (theta_x, t, size, size)
        
        Returns:
        torch.Tensor: nj values for each Floquet state (and theta_x if applicable): (size) or (theta_x, size)
        """
        if floquet_states_t.dim() == 3:
            # Case: (t, size, size)
            # Take conjugate transpose of floquet_states_t[0]
            floquet_states_conj = floquet_states_t[0].conj().T
            return torch.abs(torch.matmul(floquet_states_conj, initial_state))**2
        elif floquet_states_t.dim() == 4:
            # Case: (theta_x, t, size, size)
            print(floquet_states_t.shape)
            # Take conjugate transpose of floquet_states_t[:, 0, :, :]
            floquet_states_conj = floquet_states_t[:, 0, :, :].conj().transpose(-2, -1)
            return torch.abs(torch.einsum('xij,j->xi', floquet_states_conj, initial_state))**2
        else:
            raise ValueError("Unexpected shape for floquet_states_t")
    
    def compute_quantized_charge(self, theta_x, vdT, N_div, steps_per_segment, n=1, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        integration, floquet_states = self.simpson_integration(vdT, theta_x, N_div, steps_per_segment, n, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        init_wf = self.single_p_wf_ini()
        nj = self.nj_single_particle(init_wf, floquet_states)
        quantized_charge = torch.sum(nj * integration)
        return quantized_charge.item()
    
    def quantised_charge_single(self, vdT, N_div, steps_per_segment, max_workers=15, n=1, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        '''Preferably calculate the integral using CPU to get output integration and nj with shape (size) and calculate them for each theta_x sequentially to reduce the memory'''
        
        # Initialize the quantized charge tensor
        quantized_charge_tensor = torch.zeros(N_div + 1, device='cpu')
        
        # Generate the theta_x tensor
        thetax_tensor = torch.linspace(0, 2 * torch.pi, N_div + 1, device='cpu')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.compute_quantized_charge, theta_x, vdT, N_div, steps_per_segment, n, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
                for theta_x in thetax_tensor
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        quantized_charge_tensor = torch.tensor(results, device='cpu')
        mean_quantized_charge = torch.mean(quantized_charge_tensor)
        
        return mean_quantized_charge
    
    def plot_quantised_charge(self, vdT_tensor, N_div, steps_per_segment, max_workers=15, n=1, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=torch.pi/4, delta=None, initialise=False, fully_disorder=True):
        """
        Plot the quantized charge as a function of vdT tensor.

        Parameters:
        vdT_tensor (torch.Tensor): Tensor of vdT values.
        N_div (int): Number of divisions for theta_x.
        steps_per_segment (int): Steps per segment for time evolution.
        n, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder: Additional parameters for quantised_charge_single.
        """
        # Initialize a tensor to store the quantized charges
        quantised_charges = torch.zeros(vdT_tensor.size(0), device='cpu')

        for i, vdT in enumerate(vdT_tensor):
            charge = self.quantised_charge_single(vdT, N_div, steps_per_segment, max_workers, n, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
            quantised_charges[i] = charge
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(vdT_tensor.cpu().numpy(), quantised_charges.cpu().numpy(), marker='o', linestyle='-')
        plt.xlabel('vdT')
        plt.ylabel('Quantized Charge')
        plt.title('Quantized Charge Pumping vs. vdT Tensor')
        plt.grid(True)
        plt.show()
        
