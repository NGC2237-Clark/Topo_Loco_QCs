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