## This .py file aims to implement codes by using GPU parallel computing to accelerate the calculation.
## 1. Multi GPU parallel computing when available
## 2. Multi CPU parallel computing
## 3. Batch processing for several functions

import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool
from torch import nn


class tb_floquet_pbc_cuda(nn.Module): # Tight-binding model of square lattice with Floquet driving and periodic boundary conditions
    def __init__(self, period, lattice_constant, J_coe, num_y, num_x = 2):
        self.T = period
        # self.num_cells_y = num_cells_y
        self.ny = num_y # number of sites along the y direction
        self.nx = num_x # number of sites along the x direction
        self.a = lattice_constant
        self.J_coe = J_coe # hopping matrix element
        self.delta_AB = np.pi/(2* self.T)


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
    
    ## Exploring the bulk properties of the system
    ## Function 1. Quasienergies and states -- COMPLETED
    ## Function 2. Effective Hamiltonian and the deformation function of the bulk evolution operator -- COMPLETED
    ## Function 3. The winding number of the quasienergy gaps
    ## Function 4. The Chern number of the bulk bands
    ## Function 5. Level spacing statistics of the bulk evolution operator --COMPLETED
    ## Function 6. The Inverse Participation Ratios
    
    