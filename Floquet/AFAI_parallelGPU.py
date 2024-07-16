## This .py file aims to implement codes by using GPU parallel computing to accelerate the calculation.
## 1. Multi GPU parallel computing when available
## 2. Multi CPU parallel computing
## 3. Batch processing for several functions: vectorization of various parameters

import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool
from torch import nn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class tb_floquet_pbc_cuda(nn.Module): # Tight-binding model of square lattice with Floquet driving and periodic boundary conditions
    def __init__(self, period, lattice_constant, J_coe, num_y, num_x = 2, device=None):
        super(tb_floquet_pbc_cuda, self).__init__()
        self.T = period
        # self.num_cells_y = num_cells_y
        self.ny = num_y # number of sites along the y direction
        self.nx = num_x # number of sites along the x direction
        self.a = lattice_constant
        self.J_coe = J_coe # hopping matrix element
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
        phase = torch.exp(1j * ky * self.a).squeeze()
        # print(phase.shape)
        device = phase.device  # Use the device of the phase tensor

        # For the periodic boundary in the y direction
        if pbc == 'y' or pbc == 'xy':
            p = 0
            while 1 + 2 * p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[:, int(a), int(b)] = -J_coe_tensor * phase
                H1[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            # print(sigma_plus.shape)
            phase = phase.unsqueeze(1).unsqueeze(2).expand(-1, 2, 2)
            H1 = self.sigma_plus * (-J_coe_tensor) * phase + self.sigma_minus * (-J_coe_tensor) * phase.conj()
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
        phase = torch.exp(1j * kx * self.a).squeeze()
        device = phase.device  # Use the device of the phase tensor
        # For the periodic boundary in the x direction
        if pbc == 'x' or pbc == 'xy':
            p = 0
            while self.nx - 1 + 2 * self.nx * p < size and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                H2[:, a, b] = -J_coe_tensor * phase
                H2[:, b, a] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            phase = phase.unsqueeze(1).unsqueeze(2).expand(-1, 2, 2)
            H2 = self.sigma_plus * (-J_coe_tensor) * phase + self.sigma_minus * (-J_coe_tensor) * phase.conj()
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
        phase = torch.exp(1j * ky * self.a).squeeze()
        device = phase.device  # Use the device of the phase tensor
        # For the periodic boundary in the y direction
        if pbc == 'y' or pbc == 'xy':
            p = 0
            while 2 * p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                phase = torch.exp(1j * ky * self.a)
                phase = phase.squeeze()
                H3[:, int(a), int(b)] = -J_coe_tensor * phase
                H3[:, int(b), int(a)] = -J_coe_tensor * phase.conj()
                p += 1
        if size == 2:
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            phase = phase.unsqueeze(1).unsqueeze(2).expand(-1, 2, 2)
            H3 = self.sigma_minus * (-J_coe_tensor) * phase + self.sigma_plus * (-J_coe_tensor) * phase.conj()
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
        phase = torch.exp(1j * kx * self.a).squeeze()
        device = phase.device  # Use the device of the phase tensor
        # For the periodic boundary in the x direction
        if pbc == 'x' or pbc == 'xy':
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
            H4 = torch.zeros((batch_size, size, size), dtype=torch.cdouble, device=self.device)
            sigma_plus = self.sigma_plus.to(device)
            sigma_minus = self.sigma_minus.to(device)
            # Expand sigma_plus and sigma_minus to match the batch size
            sigma_plus = sigma_plus.unsqueeze(0).expand(batch_size, -1, -1)
            sigma_minus = sigma_minus.unsqueeze(0).expand(batch_size, -1, -1)
            phase = phase.unsqueeze(1).unsqueeze(2).expand(-1, 2, 2)
            H4 = self.sigma_minus * (-J_coe_tensor) * phase + self.sigma_plus * (-J_coe_tensor) * phase.conj()
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
        '''The time evolution operator U(t) = exp(-iH(t)) with periodic boundary conditions in the x direction and open boundary conditions in the y direction'''
        '''The n is the order of the Taylor expansion'''
        H_onsite = self.Hamiltonian_pbc_onsite(delta)
        if reverse:
            H1 = self.Hamiltonian_pbc2(kx, pbc) + H_onsite
            H2 = self.Hamiltonian_pbc1(ky, pbc) + H_onsite
            H3 = self.Hamiltonian_pbc4(kx, pbc) + H_onsite
            H4 = self.Hamiltonian_pbc3(ky, pbc) + H_onsite
            H5 = H_onsite
        else:
            H1 = self.Hamiltonian_pbc1(ky, pbc) + H_onsite
            H2 = self.Hamiltonian_pbc2(kx, pbc) + H_onsite
            H3 = self.Hamiltonian_pbc3(ky, pbc) + H_onsite
            H4 = self.Hamiltonian_pbc4(kx, pbc) + H_onsite
            H5 = H_onsite

        is_unitary = False
        while not is_unitary:
            if t < self.T/5:
                U = self.taylor_expansion(H1, t, n)
            elif self.T/5 <= t < 2 * self.T/5:
                U1 = self.taylor_expansion(H1, self.T/5, n)
                U2 = self.taylor_expansion(H2, t - self.T/5, n)
                U = U2 @ U1
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                U1 = self.taylor_expansion(H1, self.T/5, n)
                U2 = self.taylor_expansion(H2, self.T/5, n)
                U3 = self.taylor_expansion(H3, t - 2 * self.T/5, n)
                U = U3 @ U2 @ U1
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                U1 = self.taylor_expansion(H1, self.T/5, n)
                U2 = self.taylor_expansion(H2, self.T/5, n)
                U3 = self.taylor_expansion(H3, self.T/5, n)
                U4 = self.taylor_expansion(H4, t - 3 * self.T/5, n)
                U = U4 @ U3 @ U2 @ U1
            else:  # 4 * self.T/5 <= t <= self.T
                U1 = self.taylor_expansion(H1, self.T/5, n)
                U2 = self.taylor_expansion(H2, self.T/5, n)
                U3 = self.taylor_expansion(H3, self.T/5, n)
                U4 = self.taylor_expansion(H4, self.T/5, n)
                U5 = self.taylor_expansion(H5, t - 4 * self.T/5, n)
                U = U5 @ U4 @ U3 @ U2 @ U1

            U_dagger = U.conj().T
            product = U_dagger @ U
            identity = torch.eye(self.nx * self.ny, dtype=torch.cdouble, device=self.device)
            is_unitary = torch.allclose(product, identity, atol=1e-8)
            n += 1
        return U

    def taylor_expansion(self, H, t, n):
        '''Taylor expansion of exp(-iHt)'''
        U = torch.zeros_like(H, dtype=torch.cdouble, device=self.device)
        for i in range(n+1):
            U += (1/torch.tensor(float(torch.factorial(torch.tensor(i))), device=self.device)) * (-1j * t) ** i * torch.matrix_power(H, i)
        return U

    # Split operator decomposition (also known as Suzuki-Trotter decomposition)
    def infinitesimal_evol_operator(self, Hr, V_dis, dt):
        '''The infinitesimal evolution operator for the real space Hamiltonian'''
        U = torch.matrix_exp(-1j * (Hr) * dt/2) @ torch.matrix_exp(-1j * (V_dis) * dt) @ torch.matrix_exp(-1j * (Hr) * dt/2)
        return U
    
    def time_evolution_operator_pbc1(self, t, steps_per_segment, kx, ky, pbc, delta=None, reverse=False):
        '''Time evolution operator for time 0 ≤ t ≤ T with a specified number of steps per T/5 segment'''
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
    
    def quasienergy_eigenstates(self, k_num, steps_per_segment, delta=None, reverse=False, plot=False, save_path=None, pbc='x'):
        '''The quasi-energy spectrum U(kx, T) for the edge properties'''
        k_num = k_num + 1
        if pbc == 'x':
            k_x = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)
            k_y = torch.tensor([0], device=self.device, dtype=torch.float64)  # Create a tensor of zeros with the same shape as k_x
            eigenvalues_matrix = torch.zeros((k_num, self.nx * self.ny), dtype=torch.float64, device=self.device)
            wf_matrix = torch.zeros((k_num, self.nx * self.ny, self.nx * self.ny), dtype=torch.complex128, device=self.device)
            U = self.time_evolution_operator_pbc1(self.T, steps_per_segment, k_x, k_y, 'x', delta, reverse)
            eigvals, eigvecs = torch.linalg.eig(U)
            E_T = 1j * torch.log(eigvals) / self.T
            # Sort the quasienergies
            E_T, idx = E_T.real.sort(dim=1)
            # Use advanced indexing to sort eigenvectors
            eigvecs = eigvecs[torch.arange(k_num).unsqueeze(1), :, idx]
            # torch.cuda.synchronize()
            # Form the matrices of the eigenvalues and eigenvectors
            eigenvalues_matrix = E_T.real
            wf_matrix = eigvecs
            
            if plot:
                fig, ax = plt.subplots(figsize=(8, 8))
                tick_label_fontsize = 32
                label_fontsize = 34

                ax.tick_params(axis='x', labelsize=tick_label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_label_fontsize)

                ax.set_xticks([0, torch.pi/3, 2*torch.pi/3, torch.pi, 4*torch.pi/3, 5*torch.pi/3, 2*torch.pi])
                ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])

                # Print tensor properties and check for NaNs or Infs
                # print("Eigenvalues matrix shape:", eigenvalues_matrix.shape)
                # print("Eigenvalues matrix dtype:", eigenvalues_matrix.dtype)
                # print("Eigenvalues matrix (GPU):", eigenvalues_matrix)

                # Visualization without moving to CPU (only plotting will require moving to CPU)
                k_x_np = k_x.cpu().numpy()  # We need this for plotting
                eigenvalues_matrix_np = eigenvalues_matrix.cpu().numpy()  # Only move to CPU for plotting
                print(eigenvalues_matrix_np)
                for i in range(k_num):
                    ax.scatter([k_x_np[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix_np[i, :], color='black', s=0.1)
                
                ax.set_xlabel(r'$k_{x}$', fontsize=label_fontsize)
                ax.set_xlim(0, 2*torch.pi/self.a)
                ax.set_ylim(-torch.pi/self.T, torch.pi/self.T)
                
                if save_path:
                    plt.tight_layout()
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')
                plt.show()

        elif pbc == 'y':
            k_y = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)
            k_x = torch.tensor([0], device=self.device, dtype=torch.float64)  # Create a tensor of zeros with the same shape as k_x
            eigenvalues_matrix = torch.zeros((k_num, self.nx * self.ny), dtype=torch.float64, device=self.device)
            wf_matrix = torch.zeros((k_num, self.nx * self.ny, self.nx * self.ny), dtype=torch.complex128, device=self.device)
            U = self.time_evolution_operator_pbc1(self.T, steps_per_segment, k_x, k_y, 'y', delta, reverse)
            eigvals, eigvecs = torch.linalg.eig(U)
            E_T = 1j * torch.log(eigvals) / self.T
            # Sort the quasienergies
            E_T, idx = E_T.real.sort(dim=1)
            # Use advanced indexing to sort eigenvectors
            eigvecs = eigvecs[torch.arange(k_num).unsqueeze(1), :, idx]
            # Form the matrices of the eigenvalues and eigenvectors
            eigenvalues_matrix = E_T.real
            wf_matrix = eigvecs
            if plot:
                fig, ax = plt.subplots(figsize=(8, 8))
                tick_label_fontsize = 32
                label_fontsize = 34

                ax.tick_params(axis='x', labelsize=tick_label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_label_fontsize)

                ax.set_xticks([0, torch.pi/3, 2*torch.pi/3, torch.pi, 4*torch.pi/3, 5*torch.pi/3, 2*torch.pi])
                ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
                
                k_y_np = k_y.cpu().numpy()
                eigenvalues_matrix_np = eigenvalues_matrix.cpu().numpy()
                
                for i in range(k_num):
                    ax.scatter([k_y_np[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix_np[i, :], color='black', s=0.1)
                
                ax.set_xlabel(r'$k_{y}$', fontsize=label_fontsize)
                ax.set_xlim(0, 2*torch.pi/self.a)
                ax.set_ylim(-torch.pi/self.T, torch.pi/self.T)
                
                if save_path:
                    plt.tight_layout()
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')
                plt.show()
        
        elif pbc == "xy":
            k_x = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)
            k_y = torch.linspace(0, 2*torch.pi/self.a, k_num, device=self.device, dtype=torch.float64)
            eigenvalues_matrix = torch.zeros((k_num, k_num, self.nx * self.ny), dtype=torch.float64, device=self.device)
            wf_matrix = torch.zeros((k_num, k_num, self.nx * self.ny, self.nx * self.ny), dtype=torch.complex128, device=self.device)
            # print(wf_matrix.shape)
            U = self.time_evolution_operator_pbc1(self.T, steps_per_segment, k_x, k_y, 'xy', delta, reverse)
            
            # print("U computed successfully")
            eigvals, eigvecs = torch.linalg.eig(U)
            # print("Eigendecomposition successful")
            # Compute the real part of the eigenvalues for sorting
            eigv = -1j * torch.log(eigvals) / self.T
            eigv_r = eigv.real

            # Sort the eigenvalues based on their real parts
            sorted_indices = torch.argsort(eigv_r, dim=-1)

            # Reorder the eigenvalues, their real parts of the log(eigenvalues), and the eigenvectors
            sorted_eigvals = torch.gather(eigvals, -1, sorted_indices)
            sorted_eigv_r = torch.gather(eigv_r, -1, sorted_indices)
            expanded_indices = sorted_indices.unsqueeze(-2).expand_as(eigvecs)
            sorted_eigvecs = torch.gather(eigvecs, -1, expanded_indices)

            # print("everything is good", sorted_eigvecs.shape)

            # Form the tensors of the eigenvalues and eigenvectors
            eigenvalues_matrix = sorted_eigv_r.real
            wf_matrix = sorted_eigvecs
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                k_x_cpu = k_x.cpu()
                k_y_cpu = k_y.cpu()
                X, Y = torch.meshgrid(k_x_cpu, k_y_cpu, indexing='ij')
                X_np, Y_np = X.numpy(), Y.numpy()
                eigenvalues_matrix_np = eigenvalues_matrix.cpu().numpy()
                
                tick_label_fontsize = 22
                label_fontsize = 15
                
                for i in range(self.nx * self.ny):
                    ax.plot_surface(X_np, Y_np, eigenvalues_matrix_np[:, :, i], cmap='viridis')
                
                ax.set_xlabel(r'$k_{x}$', fontsize=label_fontsize)
                ax.set_ylabel(r'$k_{y}$', fontsize=label_fontsize)
                ax.set_zlabel('Quasienergy', fontsize=label_fontsize)
                ax.set_zlim(-torch.pi/self.T, torch.pi/self.T)
                ax.view_init(elev=2, azim=5)
        
        return eigenvalues_matrix, wf_matrix
    
    ## The following functions calculate the winding numbers of the quasienergy spectrum starting from obtaining the eigenvalues and eigenvectors on the grid of the quasienergy spectrum
    def eigen_grid(self, N_div, steps_per_segment, delta=None, reverse=False, plot=False, save_path=None, pbc='xy'):
        '''Version 1
        The eigenvalues and eigenvectors on the grid of the quasienergy spectrum'''
        k_x = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device)
        k_y = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device)
        t = torch.linspace(0, self.T, N_div+1, device=self.device)
        ## The eigenvalues and eigenvectors on the grid of the quasienergy spectrum
        ## The dimension of the eigenvalues_matrix is (N_kx, N_ky, N_t, nx*ny)
        ## The dimension of the wf_matrix is (N_kx, N_ky, N_t, nx*ny, nx*ny)
        U_tensor = self.time_evolution_operator_pbc1(t, steps_per_segment, k_x, k_y, pbc, delta, reverse)
        eigvals, eigvecs = torch.linalg.eig(U_tensor)
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
        sorted_eigv_r, _, _ = self.eigen_grid(N_div, steps_per_segment, delta=delta, reverse=reverse, pbc='xy')
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
    
    def animate_combined_spectra(self, N_div, steps_per_segment, delta=None, reverse=False, fps=5, filename= None):
        sorted_eigv_r, sorted_eigvals, _ = self.eigen_grid(N_div, steps_per_segment, delta=delta, reverse=reverse, pbc="xy")
        k_x = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device).cpu().numpy()
        k_y = torch.linspace(0, 2*torch.pi/self.a, N_div+1, device=self.device).cpu().numpy()
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
            ax1.set_xlabel('k_x')
            ax1.set_ylabel('k_y')
            ax1.set_zlabel('Quasienergy')
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
            
            C_p -= F_p_plus[alpha] - F_p[alpha] # Modified (Deviated) from the algorithm in the paper
        
        C_p = torch.round(C_p)
        
        if torch.sum(C_p) != 0:
            print(f"Warning: The sum of the elements in C_p is non-zero at cube ({i}, {j}, {k}).")
        
        return C_p
    
    def determine_m(self, i, j, k, eigvals, eigvecs):
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
        
        # Calculate indices for p - δ_alpha for all three directions
        indices_minus = torch.tensor([
            [(i - 1), j, k],
            [i, (j - 1), k],
            [i, j, (k - 1)]], dtype=torch.long, device=self.device)
            
        # Get eigenvalues at p - δ_alpha for all directions
        phi_p_minus_delta = eigvals[indices_minus[:, 0], indices_minus[:, 1], indices_minus[:, 2]]

        # Calculate the difference for all directions
        diff = (phi_p - phi_p_minus_delta)
        
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
            M_nu = - torch.floor((diff + torch.pi) / (2 * torch.pi))
            
            M_p[nu] = M_nu
        
        return M_p
        
    def w3(self, N_div, steps_per_segment, delta=None, reverse=False):
        eigvals, _, eigvecs = self.eigen_grid(N_div, steps_per_segment, pbc="xy")
        n_band = self.nx * self.ny
        w3 = 0
        delta = 1/N_div
        delta_space = delta * torch.pi * 2 / self.a
        delta_t = delta * self.T
        for i in range(N_div-1):
            for j in range(N_div-1): 
                for k in range(N_div-1): 
                    # print(r"($i_1, i_2, i_3$)", i+1,j+1,k+1)
                    p = torch.tensor([delta_space * (i+1), delta_space * (j+1), delta_t * (k+1)], dtype=torch.float64, device=self.device)
                    # print('p', p)
                    C_p = self.cube(i+1, j+1, k+1, N_div, eigvals, eigvecs)
                    # print(r'$C_p$', C_p)
                    M_p = self.determine_M(i+1, j+1, k+1, eigvals, C_p)
                    # print(r'$M_p$', M_p)
                    F_p = self.face_F_hat(i+1, j+1, k+1, N_div, eigvals, eigvecs)
                    m_p = self.determine_m(i+1, j+1, k+1, eigvals, eigvecs)
                    # if torch.all(C_p != 0):
                    #     print(f"($i_1, i_2, i_3$)", i+1,j+1,k+1)
                    #     print('p', p)
                    #     print(f'$C_p$', C_p)
                    #     print(f'$M_p$', M_p)
                    #     print(f'$F_p$', F_p)
                    #     print("\n")
                    # Update W3
                    w3 += torch.sum(C_p * M_p) + torch.sum(F_p * m_p)
        return w3
    
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
    
    def winding3(self, N_div, steps_per_segment, branch_cut_angle, plot=False, delta=None, reverse=False):
        w3 = self.w3(N_div, steps_per_segment, delta, reverse)
        print(w3)
        # 1. Calculate the ξ-dependent correction term for W3[Uξ].
        _, eigvals, eigvecs = self.eigen_grid(N_div, steps_per_segment, pbc="xy")
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
                p = torch.tensor([(i1+1) * 2*torch.pi/(self.a * N_div),
                                (i2+1) * 2*torch.pi/(self.a * N_div),
                                self.T], dtype=torch.float64, device=self.device)
                # print('p', p)
                # Compute F^ν_p,3
                vertices = torch.tensor([
                    [(i1+1)%N_div, (i2+1)%N_div, N_div],
                    [(i1+2)%N_div, (i2+1)%N_div, N_div],
                    [(i1+2)%N_div, (i2+2)%N_div, N_div],
                    [(i1+1)%N_div, (i2+2)%N_div, N_div]
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
                # print(F_p_3)
                # Determine K^ν_(i,j)
                K = self.determine_K(i1+1, i2+1, N_div, eigvals, eigvecs, branch_cut_angle)
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
        W3_U_xi = (w3 + correction_term) / 2
        if plot:
            # Plot for multiple branch cut angles
            plt.figure(figsize=(10, 6))
            plt.plot(branch_cut_angle.cpu().numpy(), W3_U_xi.cpu().numpy(), '-o')
            plt.xlabel('Branch Cut Angle')
            plt.ylabel('$W_{3}[U_{\\xi}]$')
            # plt.title('Winding Number vs Branch Cut Angle')
            plt.grid(True)
            plt.show()
        return W3_U_xi
        
    
class tb_floquet_tbc_cuda(nn.Module):
    def __init__(self, period, lattice_constant, J_coe, ny, nx=2, device=None):
        super(tb_floquet_tbc_cuda, self).__init__()
        self.T = period
        self.nx = nx
        self.ny = ny
        self.a = lattice_constant
        self.J_coe = J_coe
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
        H1 = torch.zeros((self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        if self.nx % 2 == 1:  # odd nx
            for i in range(self.nx * self.ny):
                a = 2 * i
                b = self.nx + 2 * i
                if b < self.nx * self.ny:
                    H1[a, b] = -J_coe_tensor
                    H1[b, a] = -J_coe_tensor.conj()
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
                while a < self.nx * self.ny and b < self.nx * self.ny:
                    H1[int(a), int(b)] = -J_coe_tensor
                    H1[int(b), int(a)] = -J_coe_tensor.conj()
                    a += 2 * self.nx
                    b += 2 * self.nx

        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 1 + 2 * p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                # Check if theta_y is already a tensor and just clone it if it's on the correct device
                if isinstance(theta_y, torch.Tensor) and theta_y.device == self.device:
                    theta_y_tensor1 = theta_y.clone().detach()
                else:
                    # If it's not a tensor or not on the correct device, properly convert it
                    theta_y_tensor1 = torch.tensor(theta_y, dtype=torch.float, device=self.device)
                H1[int(a), int(b)] = -J_coe_tensor * torch.exp(1j * theta_y_tensor1)
                H1[int(b), int(a)] = -J_coe_tensor * torch.exp(-1j * theta_y_tensor1)
                p += 1
        return H1
    
    def Hamiltonian_tbc2(self, theta_x, tbc='x'):
        '''The time-independent Hamiltonian H2 for T/5 <= t < 2T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H2 = torch.zeros((self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        n = 1
        a = -1
        b = 0
        while n <= self.ny and a < self.nx * self.ny - 1 and b < self.nx * self.ny:
            a += 2
            b += 2
            if b < n * self.nx:
                H2[a, b] = -J_coe_tensor
                H2[b, a] = -J_coe_tensor.conj()
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
                if a < self.nx * self.ny - 1 and b < self.nx * self.ny:
                    H2[a, b] = -J_coe_tensor
                    H2[b, a] = -J_coe_tensor.conj()
            if self.nx == 2:
                a += 1
                b += 1
            if b >= self.ny * self.nx - 2:
                break

        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while self.nx - 1 + 2 * self.nx * p < self.nx * self.ny and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                # Check if theta_x is already a tensor and just clone it if it's on the correct device
                if isinstance(theta_x, torch.Tensor) and theta_x.device == self.device:
                    theta_x_tensor2 = theta_x.clone().detach()
                else:
                    # If it's not a tensor or not on the correct device, properly convert it
                    theta_x_tensor2 = torch.tensor(theta_x, dtype=torch.float, device=self.device)
                H2[a, b] = -J_coe_tensor * torch.exp(1j * theta_x_tensor2)
                H2[b, a] = -J_coe_tensor * torch.exp(-1j * theta_x_tensor2)
                p += 1

        return H2
    
    def Hamiltonian_tbc3(self, theta_y, tbc='y'):
        '''The time-independent Hamiltonian H3 for 2T/5 <= t < 3T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H3 = torch.zeros((self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        if self.nx % 2 == 1:  # odd nx
            for i in range(self.nx * self.ny):
                a = 2 * i + 1
                b = self.nx + 2 * i + 1
                if b < self.nx * self.ny:
                    H3[a, b] = -J_coe_tensor
                    H3[b, a] = -J_coe_tensor.conj()
        else:  # Even nx
            n = 1
            a = 1
            b = 1 + self.nx
            if b < self.nx * self.ny:
                H3[a, b] = -J_coe_tensor
                H3[b, a] = -J_coe_tensor.conj()
            while n < self.ny and a < self.nx * self.ny - 1 and b < self.nx * self.ny - 1:
                a += 2
                b += 2
                if a < n * self.nx:
                    H3[a, b] = -J_coe_tensor
                    H3[b, a] = -J_coe_tensor.conj()
                else:
                    n += 1
                    if n % 2 == 0:  # even n
                        a -= 1
                        b -= 1
                    elif n % 2 != 0 and b < self.nx * self.ny - 1:  # odd n
                        a += 1
                        b += 1
                    else:
                        a -= 2
                        b -= 2
                    H3[a, b] = -J_coe_tensor
                    H3[b, a] = -J_coe_tensor.conj()

        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 2 * p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                # Check if theta_y is already a tensor and just clone it if it's on the correct device
                if isinstance(theta_y, torch.Tensor) and theta_y.device == self.device:
                    theta_y_tensor3 = theta_y.clone().detach()
                else:
                    # If it's not a tensor or not on the correct device, properly convert it
                    theta_y_tensor3 = torch.tensor(theta_y, dtype=torch.float, device=self.device)
                H3[int(a), int(b)] = -J_coe_tensor * torch.exp(1j * theta_y_tensor3)
                H3[int(b), int(a)] = -J_coe_tensor * torch.exp(-1j * theta_y_tensor3)
                p += 1

        return H3

    def Hamiltonian_tbc4(self, theta_x, tbc='x'):
        '''The time-independent Hamiltonian H4 for 3T/5 <= t < 4T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H4 = torch.zeros((self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)
        J_coe_tensor = torch.tensor(self.J_coe, dtype=torch.cdouble, device=self.device)

        n = 1
        a = -2
        b = -1
        while n <= self.ny and a < self.nx * self.ny - 2 and b < self.nx * self.ny - 2:
            a += 2
            b += 2
            if b < n * self.nx:
                H4[a, b] = -J_coe_tensor
                H4[b, a] = -J_coe_tensor.conj()
            else:
                n += 1
                if self.nx % 2 == 0 and self.nx != 2:  # even nx
                    a += 1
                    b += 1
                elif self.nx % 2 == 0 and self.nx == 2 and b < self.nx * self.ny - 2:  # even nx and nx = 2
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
                H4[a, b] = -J_coe_tensor
                H4[b, a] = -J_coe_tensor.conj()

        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while 2 * self.nx * (1 + p) - 1 < self.nx * self.ny and self.nx % 2 == 0:
                a = 2 * self.nx * (1 + p) - 1
                b = 2 * self.nx * p + self.nx
                # Check if theta_x is already a tensor and just clone it if it's on the correct device
                if isinstance(theta_x, torch.Tensor) and theta_x.device == self.device:
                    theta_x_tensor4 = theta_x.clone().detach()
                else:
                    # If it's not a tensor or not on the correct device, properly convert it
                    theta_x_tensor4 = torch.tensor(theta_x, dtype=torch.float, device=self.device)
                H4[a, b] = -J_coe_tensor * torch.exp(1j * theta_x_tensor4)
                H4[b, a] = -J_coe_tensor * torch.exp(-1j * theta_x_tensor4)
                p += 1

        return H4
    
    def aperiodic_Honsite(self, vd, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, contourplot=False, save_path=None):
        '''Adding aperiodic potential to the onsite Hamiltonian'''
        '''The extra phi1_ex and phi2_ex is for the convenience of adding extra phase to the potential'''
        
        # Convert vd to a tensor if it's not already one, using the recommended method
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        else:
            vd = vd.to(self.device)
        
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
        potential = torch.cos(2 * np.pi * u + phi1 + phi1_ex) + torch.cos(2 * np.pi * v + phi2 + phi2_ex)
        potential = potential.reshape(self.ny, self.nx).to(torch.cdouble)
        
        # Use broadcasting to apply the potential to all vd values
        H_aperiodic[:, sites.long(), sites.long()] = potential * -vd / 2

        H_ap = None  # Initialize H_ap to None or a default value
        if contourplot:
            # Use the first vd for plotting
            H_ap = H_aperiodic[0].diag().cpu().numpy().reshape(self.ny, self.nx).real
            plt.figure(figsize=(8, 6))
            norm = plt.Normalize(np.min(H_ap), np.max(H_ap))
            cmap = plt.get_cmap('viridis')
            plt.imshow(H_ap, cmap=cmap, norm=norm, interpolation='nearest', origin='upper')

            fontsize = 24
            ticksize = 16

            cbar = plt.colorbar(aspect=50)
            cbar.set_label(r'$V_{\mathbf{r}}T$', fontsize=fontsize)

            # Get the current tick labels
            tick_labels = cbar.ax.get_yticklabels()

            # Multiply the tick labels by self.T
            new_tick_labels = [float(tick.get_text().replace('−', '-')) * self.T for tick in tick_labels]

            # Set the tick positions and labels on the colorbar
            cbar.ax.set_yticks(cbar.ax.get_yticks())  # Set the tick positions explicitly
            cbar.ax.set_yticklabels([f'{label:.3f}' for label in new_tick_labels])

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

            return H_aperiodic, H_ap
        else:
            return H_aperiodic
    
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
    
    def Hamiltonian_disorder(self, vd, contourplot=False, initialise=False, save_path=None):
        '''The disorder Hamiltonian adding random onsite potential to the total Hamiltonian for which is uniformly distributed in the range (-vd, vd)'''
        
        # Convert vd to a tensor if it's not already one, using the recommended method
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        else:
            vd = vd.to(self.device)
            
        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)
        
        if self.H_disorder_cached is None or initialise:
            size = self.nx * self.ny
            random_values = torch.rand(size, device=self.device) * 2 - 1  # Uniform distribution between -1 and 1
            self.H_disorder_cached = torch.diag(random_values)

        # Use broadcasting to apply vd to the cached disorder matrix
        disorder_matrix = self.H_disorder_cached * vd

        if contourplot:
            # Use the first vd for plotting
            H_dis = disorder_matrix[0].diag().cpu().numpy().reshape(self.ny, self.nx)
            
            plt.figure(figsize=(8, 6))
            norm = plt.Normalize(np.min(H_dis), np.max(H_dis))
            cmap = plt.get_cmap('viridis')
            plt.imshow(H_dis, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
            fontsize = 24
            ticksize = 16
            cbar = plt.colorbar(aspect=50)
            cbar.set_label(r'$V_{\mathbf{r}}T$', fontsize=fontsize)
            
            # Get the current tick labels
            tick_labels = cbar.ax.get_yticklabels()

            # Multiply the tick labels by self.T
            new_tick_labels = [float(tick.get_text().replace('−', '-')) * self.T for tick in tick_labels]

            # Set the new tick labels on the colorbar
            cbar.ax.set_yticklabels([f'{label:.3f}' for label in new_tick_labels])

            cbar.ax.tick_params(labelsize=ticksize)
            plt.xlabel('X', fontsize=fontsize)
            plt.ylabel('Y', fontsize=fontsize)
            
            # Change font size of x and y tick labels
            x_ticks = np.arange(0, self.nx, 4)
            y_ticks = np.arange(0, self.ny, 4)
            plt.xticks(x_ticks,fontsize=ticksize)
            plt.yticks(y_ticks,fontsize=ticksize)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

        return disorder_matrix
    
    def Hamiltonian_onsite(self, vd, rotation_angle=torch.tensor(np.pi/4), a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True, contourplot=False):
        '''The time-independent Hamiltonian H5 for 4T/5 <= t < T with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        size = self.nx * self.ny

        # Convert vd to a tensor if it's not already one
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        else:
            vd = vd.to(self.device)

        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)

        # Create H_onsite with an additional dimension for batch processing
        H_onsite = torch.zeros((vd.shape[0], size, size), dtype=torch.cdouble, device=self.device)

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
            H_dis = self.Hamiltonian_disorder(vd, contourplot=contourplot, initialise=initialise)
        else:
            rotation_angle_tensor = torch.tensor(rotation_angle, device=self.device)  # Convert to tensor
            H_dis = self.aperiodic_Honsite(vd, rotation_angle_tensor, a, b, phi1_ex, phi2_ex, contourplot=contourplot)

        H5 = H_onsite + H_dis

        return H5
    
    def Hamiltonian_tbc(self, t, tbc, vd, rotation_angle, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        """The Hamiltonian H(t) with twisted boundary conditions in either x, y, or both x and y directions in the real space """
        
        # Ensure t is within [0, T)
        t = t % self.T

        # Convert vd to a tensor if it's not already one
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        else:
            vd = vd.to(self.device)

        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)

        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)

        # Assume Hamiltonian_tbc1, Hamiltonian_tbc2, Hamiltonian_tbc3, Hamiltonian_tbc4 return tensors
        # that can be broadcast with H_onsite
        if t < self.T/5:
            H_tbc = self.Hamiltonian_tbc1(theta_y, tbc)
        elif self.T/5 <= t < 2 * self.T/5:
            H_tbc = self.Hamiltonian_tbc2(theta_x, tbc)
        elif 2 * self.T/5 <= t < 3 * self.T/5:
            H_tbc = self.Hamiltonian_tbc3(theta_y, tbc)
        elif 3 * self.T/5 <= t < 4 * self.T/5:
            H_tbc = self.Hamiltonian_tbc4(theta_x, tbc)
        else:  # 4 * self.T/5 <= t < self.T
            H_tbc = 0

        # Add H_tbc to all elements in the batch
        if H_tbc != 0:
            H = H_tbc + H_onsite
        else:
            H = H_onsite

        return H
    
    def time_evolution_operator(self, t, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        '''The time evolution operator U(t) = exp(-iH(t))
        n is the order of expansion of the time evolution operator'''
        
        # Convert vd to a tensor if it's not already one
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        else:
            vd = vd.to(self.device)

        # Reshape vd for broadcasting
        vd = vd.reshape(-1, 1, 1)

        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
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
    
    def time_evolution_operator1(self, t, steps_per_segment, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):
        '''Time evolution operator for time t ≤ T with a specified number of steps per T/5 segment'''
        
        # Handle both scalar and tensor vd
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Ensure vd is a 3D tensor
        if vd.dim() == 1:
            vd = vd.unsqueeze(1).unsqueeze(2)
        elif vd.dim() == 2:
            vd = vd.unsqueeze(2)
        
        batch_size = vd.shape[0]
        
        # Calculate dt based on the number of steps per segment
        dt = self.T / (5 * steps_per_segment)
        
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_x, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H2 = self.Hamiltonian_tbc2(theta_y, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H3 = self.Hamiltonian_tbc3(theta_x, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H4 = self.Hamiltonian_tbc4(theta_y, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        
        U = torch.eye(self.nx * self.ny, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        total_steps = int(t / dt)
        
        for step in range(total_steps):
            current_t = step * dt
            if current_t < self.T/5:
                Hr = H1
            elif current_t < 2*self.T/5:
                Hr = H2
            elif current_t < 3*self.T/5:
                Hr = H3
            elif current_t < 4*self.T/5:
                Hr = H4
            else:
                Hr = torch.zeros_like(H1)
            
            U_step = self.infinitesimal_evol_operator(Hr, H_onsite, dt)
            U = U_step @ U
        
        # Handle any remaining time
        remaining_time = t - total_steps * dt
        if remaining_time > 0:
            if t < self.T/5:
                Hr = H1
            elif t < 2*self.T/5:
                Hr = H2
            elif t < 3*self.T/5:
                Hr = H3
            elif t < 4*self.T/5:
                Hr = H4
            else:
                Hr = torch.zeros_like(H1)
            
            U_step = self.infinitesimal_evol_operator(Hr, H_onsite, remaining_time)
            U = U_step @ U
        
        return U
    
    ## Exploring the bulk properties of the system
    ## Function 1. Quasienergies and states -- COMPLETED
    ## Function 2. Effective Hamiltonian and the deformation function of the bulk evolution operator -- COMPLETED
    ## Function 3. The Winding number of the quasienergy gaps
    ## Function 4. The Chern number of the bulk bands
    ## Function 5. Level spacing statistics of the bulk evolution operator --COMPLETED
    ## Function 6. The Inverse Participation Ratios
    
    def quasienergies_states_bulk(self, vd, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True):
        """The quasi-energy spectrum for the bulk U(theta_x, theta_y, T) properties"""
        
        # Ensure vd is a tensor
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Reshape vd to (batch_size, 1, 1) for broadcasting
        vd = vd.reshape(-1, 1, 1)
        
        U = self.time_evolution_operator(self.T, 'xy', vd, rotation_angle, theta_x, theta_y, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        
        # Perform eigendecomposition for each matrix in the batch
        eigvals, eigvecs = torch.linalg.eig(U)
        
        E_T = torch.log(eigvals).imag / self.T
        
        # Sort the quasienergies for each element in the batch
        E_T_cpu = E_T.cpu()
        
        # Get the sorting indices for each batch element
        idx = E_T_cpu.argsort(dim=-1)
        
        # Use advanced indexing to sort E_T and eigvecs
        batch_size = E_T.shape[0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, E_T.shape[1])
        
        E_T_sorted = E_T[batch_indices, idx].to(self.device)
        eigvecs_sorted = eigvecs[batch_indices.unsqueeze(-1).expand(-1, -1, eigvecs.shape[-1]), 
                                idx.unsqueeze(-1).expand(-1, -1, eigvecs.shape[-1])].to(self.device)
        
        return E_T_sorted.real, eigvecs_sorted
    
    def avg_level_spacing_bulk(self, vd, theta_x=0, theta_y=0, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The level spacing statistics of the bulk evolution operator for a batch of vd values'''
        
        # Ensure vd is a tensor
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Reshape vd to (batch_size, 1, 1) for broadcasting
        vd = vd.reshape(-1, 1, 1)
        
        E_T, _ = self.quasienergies_states_bulk(vd, theta_x, theta_y, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        
        # Compute level spacing for each batch element
        delta = torch.diff(E_T, dim=1)
        level_spacing = torch.minimum(delta[:, 1:], delta[:, :-1]) / torch.maximum(delta[:, 1:], delta[:, :-1])
        level_spacing_avg = level_spacing.mean(dim=1)
        
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            tick_label_fontsize = 32
            label_fontsize = 34
            x_vals = vd.squeeze() * self.T
            ax.scatter(x_vals.cpu().numpy(), level_spacing_avg.cpu().numpy(), c='b')
            ax.set_xlabel(r'Aperiodic potential, $\delta V_{d}$T', fontsize=label_fontsize)
            ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=label_fontsize)
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
        
        return level_spacing_avg
    
    def avg_LSR__disorder_realisation(self, vd_min, vd_max, vd_num, N_dis, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of disorder realisation"""
        vd = torch.linspace(vd_min, vd_max, vd_num, device=self.device)
        avg = torch.zeros(vd_num, device=self.device)

        for _ in range(N_dis):
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            avg_single = self.avg_level_spacing_bulk(vd, delta=delta, fully_disorder=True)
            avg += avg_single
            torch.cuda.empty_cache()

        avg_LSR = avg / N_dis
        avg_LSR_cpu = avg_LSR.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd_plot = vd.cpu().numpy() * self.T
        ax.scatter(vd_plot, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Disorder strength, $\delta V_{d}$T', fontsize=label_fontsize)
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
    
    def avg_LSR_phase_realisation(self, vd_min, vd_max, vd_num, N_phi, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of phase realisation"""
        vd = torch.linspace(vd_min, vd_max, vd_num, device=self.device)
        avg = torch.zeros(vd_num, device=self.device)
        phi1_vals = torch.rand(N_phi, device=self.device) * 2 * np.pi
        phi2_vals = torch.rand(N_phi, device=self.device) * 2 * np.pi

        for i in range(N_phi):
            avg_single = self.avg_level_spacing_bulk(vd, phi1_ex=phi1_vals[i], phi2_ex=phi2_vals[i], 
                                                    delta=delta, fully_disorder=False)
            avg += avg_single
            torch.cuda.empty_cache()

        avg_LSR = avg / N_phi
        avg_LSR_cpu = avg_LSR.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd_plot = vd.cpu().numpy() * self.T
        ax.scatter(vd_plot, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Quasiperiodic potential strength, $\delta V_{d}$T', fontsize=label_fontsize)
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
    ## Function 2. Deformed time-periodic evolution operator
    ## Function 3. Disordered-averaged transmission probability --COMPLETED
    ## Function 4. The Inverse Participation Ratios
    
    def quasienergies_states_edge(self, vd, rotation_angle, theta_x_num, a=0, b=0, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The quasi-energy spectrum for the edge U(kx, T) properties'''
        '''The output should be the quasienergies which are the diagonal elements of the effective Hamiltonian after diagonalisation. This is an intermediate step towards the 'deformed' time-periodic evolution operator '''
        
        # Ensure vd is a tensor
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        vd = vd.reshape(-1, 1, 1)  # Reshape for broadcasting
        
        theta_x = torch.linspace(0, 2 * torch.pi, theta_x_num, device=self.device)
        
        batch_size = vd.shape[0]
        eigenvalues_matrix = torch.zeros((batch_size, theta_x_num, self.nx * self.ny), device=self.device)
        wf_matrix = torch.zeros((batch_size, theta_x_num, self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)

        for i_index, i in enumerate(theta_x):
            U = self.time_evolution_operator(self.T, 'x', vd, rotation_angle, theta_x=i, a=a, b=b, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
            eigvals, eigvecs = torch.linalg.eig(U)
            E_T = torch.log(eigvals).imag / self.T
            
            # Sort for each batch
            E_T, idx = E_T.sort(dim=-1)
            eigvecs = torch.gather(eigvecs, -1, idx.unsqueeze(-2).expand_as(eigvecs))
            
            eigenvalues_matrix[:, i_index, :] = E_T
            wf_matrix[:, i_index, :, :] = eigvecs

        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            tick_label_fontsize = 32
            label_fontsize = 34

            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=tick_label_fontsize)

            theta_x_cpu = theta_x.cpu().numpy()
            
            ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
            ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
            
            # Plot for the first vd in the batch
            eigenvalues_matrix_cpu = eigenvalues_matrix[0].cpu().numpy()
            for i in range(theta_x_num):
                ax.scatter([theta_x_cpu[i]] * eigenvalues_matrix.shape[2], eigenvalues_matrix_cpu[i], c='b', s=0.1)

            ax.set_xlabel(r'$\theta_{x}$', fontsize=label_fontsize)
            ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(-np.pi / self.T, np.pi / self.T)

            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')

            plt.show()

        return eigenvalues_matrix, wf_matrix
    
    def taylor_expansion_single(self, H, t, i):
        '''Compute a single term in the Taylor expansion'''
        return 1 / np.math.factorial(i) * (-1j * t) ** i * torch.matrix_power(H, i)
    
    def taylor_expansion(self, H, t, n):
        '''Taylor expansion of exp(-iHt) using multiple threads'''
        U = torch.zeros_like(H, dtype=torch.complex128)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.taylor_expansion_single, H, t, i) for i in range(n + 1)]
            
            for future in futures:
                U += future.result()
        
        return U

    def time_evolution_1period(self, n, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):
        # Ensure vd is a tensor
        if not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        vd = vd.reshape(-1, 1, 1)

        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_x, tbc).unsqueeze(0) + H_onsite
        H2 = self.Hamiltonian_tbc2(theta_y, tbc).unsqueeze(0) + H_onsite
        H3 = self.Hamiltonian_tbc3(theta_x, tbc).unsqueeze(0) + H_onsite
        H4 = self.Hamiltonian_tbc4(theta_y, tbc).unsqueeze(0) + H_onsite
        H5 = H_onsite

        batch_size = vd.shape[0]
        U_batch = []

        for b in range(batch_size):
            H11 = H1[b].cpu()
            H22 = H2[b].cpu()
            H33 = H3[b].cpu()
            H44 = H4[b].cpu()
            H55 = H5[b].cpu()
            
            is_unitary = False
            n_local = n
            while not is_unitary:
                U1 = self.taylor_expansion(H11, self.T / 5, n_local)
                U2 = self.taylor_expansion(H22, self.T / 5, n_local)
                U3 = self.taylor_expansion(H33, self.T / 5, n_local)
                U4 = self.taylor_expansion(H44, self.T / 5, n_local)
                U5 = self.taylor_expansion(H55, self.T / 5, n_local)
                U = U5 @ U4 @ U3 @ U2 @ U1
                U_dagger = U.conj().T
                product = U @ U_dagger
                identity = torch.eye(self.nx * self.ny, dtype=torch.complex128)
                is_unitary = torch.allclose(product, identity, atol=1e-10)
                print(n_local)
                n_local += 1
            
            U_batch.append(U)

        return torch.stack([u.to(self.device) for u in U_batch])
    
    def infinitesimal_evol_operator(self, Hr, V_dis, dt):
        '''The infinitesimal evolution operator for the real space Hamiltonian'''
        U = torch.matrix_exp(-1j * (Hr) * dt/2) @ torch.matrix_exp(-1j * (V_dis) * dt) @ torch.matrix_exp(-1j * (Hr) * dt/2)
        return U
    
    def time_evol_op(self, N, steps_per_segment, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):
        '''Time evolution operator over N periods (t = N * T) with specified steps per T/5 segment for time-periodic Hamiltonian'''
        # Handle both scalar and tensor vd
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Ensure vd is a 3D tensor
        if vd.dim() == 1:
            vd = vd.unsqueeze(1).unsqueeze(2)
        elif vd.dim() == 2:
            vd = vd.unsqueeze(2)
        
        batch_size = vd.shape[0]
        
        # Calculate dt based on steps_per_segment
        dt = self.T / (5 * steps_per_segment)
        
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_x, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H2 = self.Hamiltonian_tbc2(theta_y, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H3 = self.Hamiltonian_tbc3(theta_x, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        H4 = self.Hamiltonian_tbc4(theta_y, tbc).unsqueeze(0).expand(batch_size, -1, -1)
        
        def evolve_one_period():
            print("Evolve one period")
            U_period = torch.eye(self.nx * self.ny, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
            print("steps_per_segment", steps_per_segment)

            # Stack all Hamiltonians into a single tensor
            H_stack = torch.stack([torch.zeros_like(H1), H4, H3, H2, H1], dim=0)

            for _ in range(steps_per_segment):
                # Apply infinitesimal_evol_operator to all Hamiltonians at once
                U_steps = self.infinitesimal_evol_operator(H_stack, H_onsite.unsqueeze(0).expand(5, -1, -1, -1), dt)
                
                # Multiply U_period with all U_steps
                for U_step in U_steps:
                    U_period = U_step @ U_period

            return U_period

        # Evolve for N periods
        U_one_period = evolve_one_period()
        U = torch.matrix_power(U_one_period, N)

        return U
    
    def real_time_trans(self, N_times, steps_per_segment, initial_position, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        """The real time transmission Amplitude of evolved initial wavepacket at given initial_position (input) over N_times period"""
        U = self.time_evol_op(N_times, steps_per_segment, tbc, vd, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)
        
        batch_size = U.shape[0]
        num_sites = U.shape[1]
        
        # Create initial state vector for each batch
        vector = torch.zeros((batch_size, num_sites), dtype=torch.complex128, device=self.device)
        vector[:, initial_position - 1] = 1.0
        
        # Perform batch matrix-vector multiplication
        Ua = torch.bmm(U, vector.unsqueeze(-1)).squeeze(-1)
        
        return Ua
    
    def transmission_prob(self, N_max, steps_per_segment, initial_position, energy, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        # Ensure vd is a tensor
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Ensure energy is a tensor
        if isinstance(energy, (int, float)):
            energy = torch.tensor([energy], device=self.device)
        elif not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=self.device)
        
        # Ensure phi1 and phi2 are tensors
        if isinstance(phi1, (int, float)):
            phi1 = torch.tensor([phi1], device=self.device)
        elif not isinstance(phi1, torch.Tensor):
            phi1 = torch.tensor(phi1, device=self.device)
        
        if isinstance(phi2, (int, float)):
            phi2 = torch.tensor([phi2], device=self.device)
        elif not isinstance(phi2, torch.Tensor):
            phi2 = torch.tensor(phi2, device=self.device)
        
        vd_size = vd.shape[0]
        energy_size = energy.shape[0]
        phi_size = max(phi1.shape[0], phi2.shape[0])
        num_sites = self.nx * self.ny
        
        # Reshape vd, energy, phi1, and phi2 for broadcasting
        vd = vd.view(vd_size, 1, 1, 1)
        energy = energy.view(1, energy_size, 1, 1)
        phi1 = phi1.view(1, 1, phi_size, 1)
        phi2 = phi2.view(1, 1, phi_size, 1)
        
        # Compute U for one period
        U_one_period = self.time_evol_op(1, steps_per_segment, tbc, vd, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)
        
        # Compute powers of U_one_period for all required periods
        U_powers = [torch.eye(num_sites, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(vd_size, phi_size, -1, -1)]
        for _ in range(N_max):
            U_powers.append(torch.matmul(U_powers[-1], U_one_period))
        
        # Stack all U_powers
        U_all_periods = torch.stack(U_powers)
        
        # Create initial state vector
        vector = torch.zeros((vd_size, phi_size, num_sites), dtype=torch.complex128, device=self.device)
        vector[:, :, initial_position - 1] = 1.0
        
        # Compute G for all periods at once
        G_all = torch.matmul(U_all_periods, vector.unsqueeze(-1)).squeeze(-1)
        G_all = G_all.view(N_max + 1, vd_size, phi_size, num_sites)
        
        # Compute complex exponentials for all periods at once
        nn = torch.arange(N_max + 1, device=self.device)
        complex_exponent = 1j * energy * nn.unsqueeze(1) * self.T
        
        # Compute G_aa
        G_aa = torch.sum(G_all.unsqueeze(1) * torch.exp(complex_exponent).unsqueeze(-1).unsqueeze(-1), dim=0) / (N_max + 1)
        
        transmission_prob = torch.abs(G_aa)**2
        
        return transmission_prob  # shape: (vd_size, energy_size, phi_size, num_sites)
    
    ## The following two functions "real_trans_prob_avg_dis" and "trans_prob_avg_disorder_realisation" only deal with the fully disordered case
    def real_trans_prob_avg_dis(self, N_dis, N_times, steps_per_segment, initial_pos, tbc, vd, rotation_angle=0, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None):
        '''The disorder averaged real-time transmission probability of the evolved initial wavepacket at given initial_position (input)
        over N_times period averaging over N_dis realisation'''
        '''The first initial_position is 1 instead of 0'''
        
        # Ensure vd is a tensor
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        batch_size = vd.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((batch_size, num_sites), device=self.device)
        
        for N in range(N_dis):
            self.H_disorder_cached = None
            print(N)
            
            real_amp = self.real_time_trans(N_times, steps_per_segment, initial_pos, tbc, vd, rotation_angle, 
                                            theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=True)
            
            avg += torch.abs(real_amp)**2
            
            del real_amp
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_dis
        
        del avg
        torch.cuda.empty_cache()
        
        return avg_tp   # shape: a 3D tensor with dimensions (vd_size, energy_size, total number of sites)
    
    def trans_prob_avg_disorder_realisation(self, N_dis, N_max, steps_per_segment, initial_position, energy, tbc, vd, rotation_angle=0, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None):
        """The disorder averaged transmission probability of the evolved initial wavepacket at given initial_position (input) over N_max period averaging over N_dis realisation"""
        '''The first initial_position is 1 instead of 0'''
        '''The output should be a tensor with dimensions (vd_size, energy_size, total numbers of sites)'''
        
        # Ensure vd is a tensor
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        # Ensure energy is a tensor
        if isinstance(energy, (int, float)):
            energy = torch.tensor([energy], device=self.device)
        elif not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=self.device)
        
        vd_size = vd.shape[0]
        energy_size = energy.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((vd_size, energy_size, num_sites), device=self.device)
        
        for N in range(N_dis):
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            print(N)
            
            trans_prob_single = self.transmission_prob(N_max, steps_per_segment, initial_position, energy, tbc, vd, 
                                                    rotation_angle, theta_x, theta_y, a, b, phi1, phi2, 
                                                    delta, fully_disorder=True)
            # Explicitly squeeze out the singleton dimension
            # trans_prob_single = trans_prob_single.squeeze(2)
            
            avg += trans_prob_single  # Sum up the averages
            
            del trans_prob_single  # Free up memory of the single average once added
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_dis
        
        del avg
        torch.cuda.empty_cache()
        
        return avg_tp
    
    ## The following two functions "real_trans_prob_avg_phase" and "trans_prob_avg_phase_realisation" only deal with the aperiodic case
    def real_trans_prob_avg_phase(self, N_phi, N_times, steps_per_segment, initial_pos, tbc, vd, rotation_angle=np.pi/4, theta_x=0, theta_y=0, a=0, b=0, delta=None):
        # Ensure vd is a tensor
        if isinstance(vd, (int, float)):
            vd = torch.tensor([vd], device=self.device)
        elif not isinstance(vd, torch.Tensor):
            vd = torch.tensor(vd, device=self.device)
        
        batch_size = vd.shape[0]
        num_sites = self.nx * self.ny
        
        avg = torch.zeros((batch_size, num_sites), device=self.device)
        phi1_vals = np.random.uniform(0, 2*np.pi, N_phi)
        phi2_vals = np.random.uniform(0, 2*np.pi, N_phi)
        
        for N in range(N_phi):
            print(N)
            real_amp = self.real_time_trans(N_times, steps_per_segment, initial_pos, tbc, vd, rotation_angle, 
                                            theta_x, theta_y, a, b, phi1=phi1_vals[N], phi2=phi2_vals[N], 
                                            delta=delta, fully_disorder=False)
            avg += torch.abs(real_amp)**2
            del real_amp
            torch.cuda.empty_cache()
        
        avg_tp = avg / N_phi
        del avg
        torch.cuda.empty_cache()
        return avg_tp

    def trans_prob_avg_phase_realisation(self, N_phi, N_max, steps_per_segment, initial_pos, energy, tbc, vd, rotation_angle=np.pi/4, theta_x=0, theta_y=0, a=0, b=0, delta=None):
        # Ensure vd and energy are tensors
        vd = torch.tensor(vd, device=self.device) if not isinstance(vd, torch.Tensor) else vd
        energy = torch.tensor(energy, device=self.device) if not isinstance(energy, torch.Tensor) else energy
        
        vd_size, energy_size = vd.shape[0], energy.shape[0]
        num_sites = self.nx * self.ny

        # Generate all phi1 and phi2 values at once
        phi1_vals = torch.from_numpy(np.random.uniform(0, 2*np.pi, N_phi)).to(self.device)
        phi2_vals = torch.from_numpy(np.random.uniform(0, 2*np.pi, N_phi)).to(self.device)

        # Expand dimensions for broadcasting
        vd = vd.view(vd_size, 1, 1, 1)
        energy = energy.view(1, energy_size, 1, 1)
        phi1_vals = phi1_vals.view(1, 1, N_phi, 1)
        phi2_vals = phi2_vals.view(1, 1, N_phi, 1)

        # Compute transmission probabilities for all combinations in one go
        trans_prob_all = self.transmission_prob(N_max, steps_per_segment, initial_pos, energy, tbc, vd, rotation_angle, 
                                                theta_x, theta_y, a, b, phi1=phi1_vals, phi2=phi2_vals, 
                                                delta=delta, fully_disorder=False)

        # Average over the phi dimension
        avg_tp = torch.mean(trans_prob_all, dim=2)

        return avg_tp
    
    
    
    