import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class tb_floquet_tbc_cuda:
    def __init__(self, period, lattice_constant, J_coe, ny, nx=2, device='cuda'):
        self.T = period
        self.nx = nx
        self.ny = ny
        self.a = lattice_constant
        self.J_coe = J_coe
        self.delta_AB = np.pi / (2 * self.T)
        self.H_disorder_cached = None  # Initialize H_disorder_cached as None
        self.device = torch.device(device)

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
                # print(int(a), int(b), H1[int(a), int(b)])
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
        size = self.nx * self.ny
        H_aperiodic = torch.zeros((size, size), dtype=torch.cdouble, device=self.device)
        sites = self.lattice_numbering()

        # Ensure the rotation angle is on the right device
        rotation_angle = rotation_angle.to(self.device) if isinstance(rotation_angle, torch.Tensor) else torch.tensor(rotation_angle, device=self.device)

        # Computing u and v, ensure they are tensors of appropriate dimensions
        # x_indices = torch.arange(self.nx, device=self.device).reshape(1, -1)
        x_indices = torch.linspace(-(self.nx-1)/2, (self.nx-1)/2, steps=self.nx, device=self.device).reshape(1, -1)
        # print(x_indices)
        # y_indices = torch.arange(self.ny, device=self.device).reshape(-1, 1)
        y_indices = torch.linspace(-(self.ny-1)/2, (self.ny-1)/2, steps=self.ny, device=self.device).reshape(-1, 1)
        # print(y_indices)
        u = x_indices * torch.cos(rotation_angle) - y_indices * torch.sin(rotation_angle)
        v = x_indices * torch.sin(rotation_angle) + y_indices * torch.cos(rotation_angle)

        # Compute phase shifts using proper tensor operations
        phi1 = 2 * np.pi * (a * torch.cos(rotation_angle).item() - b * torch.sin(rotation_angle).item())
        phi2 = 2 * np.pi * (a * torch.sin(rotation_angle).item() + b * torch.cos(rotation_angle).item())
        # print(phi1, phi2)
        # Calculate the potential and assign it correctly to the diagonal
        potential = torch.cos(2 * np.pi * u + phi1 + phi1_ex) + torch.cos(2 * np.pi * v + phi2 + phi2_ex)
        potential = potential.reshape(self.ny, self.nx).to(torch.cdouble)
        H_aperiodic[sites.long(), sites.long()] = potential * -vd/2
        # print(H_aperiodic)
        H_ap = None  # Initialize H_ap to None or a default value
        if contourplot:
            H_ap = H_aperiodic.diag().cpu().numpy().reshape(self.ny, self.nx).real
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
        if self.H_disorder_cached is None or initialise:
            size = self.nx * self.ny
            random_values = torch.rand(size, device=self.device) * 2 - 1  # Uniform distribution between -1 and 1
            self.H_disorder_cached = torch.diag(random_values)

        disorder_matrix = self.H_disorder_cached * vd

        if contourplot:
            # Ensure that the disorder matrix is properly reshaped into a 2D grid matching the lattice
            # Check if the disorder matrix is a square matrix of size (size x size)
            if disorder_matrix.shape[0] != self.nx * self.ny:
                raise ValueError(f"Expected disorder matrix to be square with size {self.nx * self.ny}")

            # Convert tensor to numpy for plotting and reshape
            H_dis = disorder_matrix.diag().cpu().numpy().reshape(self.ny, self.nx)
            
            plt.figure(figsize=(8, 6))
            # Normalize the colormap to fit the range of potential energies
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

        if fully_disorder:
            H_dis = self.Hamiltonian_disorder(vd, contourplot=contourplot, initialise=initialise)
        else:
            rotation_angle_tensor = torch.tensor(rotation_angle, device=self.device)  # Convert to tensor
            H_dis = self.aperiodic_Honsite(vd, rotation_angle_tensor, a, b, phi1_ex, phi2_ex, contourplot=contourplot)
        H5 = H_onsite + H_dis
        return H5

    def Hamiltonian_tbc(self, t, tbc, vd, rotation_angle, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        """The Hamiltonian H(t) with twisted boundary conditions in either x, y, or both x and y directions in the real space """
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)

        if t < self.T/5:
            H = self.Hamiltonian_tbc1(theta_y, tbc) + H_onsite
        elif self.T/5 <= t < 2 * self.T/5:
            H = self.Hamiltonian_tbc2(theta_x, tbc) + H_onsite
        elif 2 * self.T/5 <= t < 3 * self.T/5:
            H = self.Hamiltonian_tbc3(theta_y, tbc) + H_onsite
        elif 3 * self.T/5 <= t < 4 * self.T/5:
            H = self.Hamiltonian_tbc4(theta_x, tbc) + H_onsite
        elif 4 * self.T/5 <= t < self.T:
            H = H_onsite

        return H

    def time_evolution_operator(self, t, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1_ex=0, phi2_ex=0, delta=None, initialise=False, fully_disorder=True):
        '''The time evolution operator U(t) = exp(-iH(t))
        n is the order of expansion of the time evolution operator'''
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_y, tbc) + H_onsite
        # print("H1", H1)
        H2 = self.Hamiltonian_tbc2(theta_x, tbc) + H_onsite
        # print("H2", H2)
        H3 = self.Hamiltonian_tbc3(theta_y, tbc) + H_onsite
        # print("H3", H3)
        H4 = self.Hamiltonian_tbc4(theta_x, tbc) + H_onsite
        # print("H4", H4)
        H5 = H_onsite
        # print("H5", H5)

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

    def quasienergies_states_bulk(self, vd, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True):
        """The quasi-energy spectrum for the bulk U(theta_x, theta_y, T) properties"""
        U = self.time_evolution_operator(self.T, 'xy', vd, rotation_angle, theta_x, theta_y, a, b, phi1_ex, phi2_ex, delta, initialise, fully_disorder)
        eigvals, eigvecs = torch.linalg.eig(U)
        E_T = torch.log(eigvals).imag / self.T
        # Sort the quasienergies on CPU
        E_T_cpu = E_T.cpu()
        idx = E_T_cpu.argsort()
        E_T = E_T[idx].to(self.device)
        eigvecs = eigvecs[:, idx].to(self.device)
        return E_T.real, eigvecs
    
    def avg_level_spacing_bulk(self, vd, theta_x, theta_y, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True):
        '''The level spacing statistics of the bulk evolution operator'''
        E_T, _ = self.quasienergies_states_bulk(vd, theta_x, theta_y, a, b, phi1_ex, phi2_ex, rotation_angle, delta, initialise, fully_disorder)
        delta = torch.diff(E_T)
        level_spacing = torch.minimum(delta[1:], delta[:-1]) / torch.maximum(delta[1:], delta[:-1])
        level_spacing_avg = level_spacing.mean()
        return level_spacing_avg.item()
    
    def avg_level_spacing_bulk_vd(self, vd_min, vd_max, vd_num, a=0, b=0, phi1_ex=0, phi2_ex=0, rotation_angle=np.pi/4, delta=None, fully_disorder=True, plot=True, save_path=None):
        '''The average level spacing statistics of the bulk evolution operator as the function of the disorder strength vd for fixed value of flux theta_x and theta_y'''
        vd = torch.linspace(vd_min, vd_max, vd_num, device=self.device)
        avg_level_spacings = torch.zeros_like(vd)
        x_vals = torch.zeros_like(vd)
        for i in range(vd_num):
            avg_level_spacing = self.avg_level_spacing_bulk(vd[i].item(), theta_x = 0, theta_y=0, a=a, b=b, phi1_ex=phi1_ex, phi2_ex=phi2_ex, rotation_angle=rotation_angle, delta=delta, initialise=False, fully_disorder=fully_disorder)
            avg_level_spacings[i] = avg_level_spacing  # Assign value using indexing
            x_vals[i] = vd[i].item() * self.T  # Assign value using indexing
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            tick_label_fontsize = 32
            label_fontsize = 34
            ax.scatter(x_vals.cpu().numpy(), avg_level_spacings.cpu().numpy(), c='b')
            ax.set_xlabel(r'Aperiodic potential, $\delta V_{d}$T', fontsize=label_fontsize)
            ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=label_fontsize)
            if save_path:
                plt.tight_layout()
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
        return avg_level_spacings
    
    def avg_LSR__disorder_realisation(self, vd_min, vd_max, vd_num, N_dis, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of disorder realisation"""
        '''This function is mainly for the fully disordered case'''
        N = 0
        avg = torch.zeros(vd_num, device=self.device)  # Initialize the avg tensor on the GPU

        while N < N_dis:
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            print(N)
            avg_single = self.avg_level_spacing_bulk_vd(vd_min, vd_max, vd_num, delta=delta, fully_disorder=True, plot=False)
            avg += avg_single  # Sum up the averages
            del avg_single  # Free up memory of the single average once added
            torch.cuda.empty_cache()  # Clear unused memory
            N += 1
        avg_LSR = avg / N_dis  # Compute the mean average level-spacing ratio
        avg_LSR_cpu = avg_LSR.cpu().numpy()  # Transfer the final average to CPU for plotting
        del avg  # Delete the GPU tensor
        torch.cuda.empty_cache()  # Clear GPU cache

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd = torch.linspace(vd_min * self.T, vd_max * self.T, vd_num).numpy()  # Generate vd on CPU directly
        ax.scatter(vd, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Disorder strength, $\delta V_{d}$T', fontsize=label_fontsize)
        ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.axhline(y=0.386, color='r', linestyle='--', label='Poisson')
        ax.axhline(y=0.5996, color='g', linestyle='--', label='GUE')
        ax.axhline(y=0.53, color='black', linestyle='--', label='GOE')
        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
        return avg_LSR

    def avg_LSR_phase_realisation(self, vd_min, vd_max, vd_num, N_phi, delta=None, save_path=None):
        """Plot the average level-spacing ratio of the bulk time evolution operator averaging over given number of phase realisation"""
        '''This function is mainly for the aperiodic case'''
        avg = torch.zeros(vd_num, device=self.device) # Initialize the avg tensor on the GPU
        phi1_vals = np.random.uniform(0, 2*np.pi, N_phi)
        phi2_vals = np.random.uniform(0, 2*np.pi, N_phi)
        for i in range(N_phi):
            print(i)
            avg_single = self.avg_level_spacing_bulk_vd(vd_min, vd_max, vd_num, phi1_ex=phi1_vals[i], phi2_ex=phi2_vals[i], \
                                                 delta=delta, fully_disorder=False, plot=False)
            avg += avg_single
            del avg_single
            torch.cuda.empty_cache()
        avg_LSR = avg / N_phi
        avg_LSR_cpu = avg_LSR.cpu().numpy()
        del avg
        torch.cuda.empty_cache()

        fig, ax = plt.subplots(figsize=(12, 8))
        tick_label_fontsize = 32
        label_fontsize = 34
        vd = torch.linspace(vd_min * self.T, vd_max * self.T, vd_num).numpy()  # Generate vd on CPU directly
        ax.scatter(vd, avg_LSR_cpu, c='b')
        ax.set_xlabel(r'Quasiperiodic potential strength, $\delta V_{d}$T', fontsize=label_fontsize)
        ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.axhline(y=0.386, color='r', linestyle='--', label='Poisson')
        ax.axhline(y=0.53, color='black', linestyle='--', label='GOE')
        ax.axhline(y=0.5996, color='g', linestyle='--', label='GUE')

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

    def quasienergies_states_edge(self, vd, rotation_angle, theta_num, tbc='x', a=0, b=0, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The quasi-energy spectrum for the edge U(kx, T) properties'''
        '''The output should be the quasienergies which are the diagonal elements of the effective Hamiltonian after diagonalisation. This is an intermeidate step towards the 'deformed' time-periodic evolution operator '''
        theta_x = torch.linspace(0, 2 * torch.pi, theta_num, device=self.device)
        theta_y = torch.linspace(0, 2 * torch.pi, theta_num, device=self.device)
        eigenvalues_matrix = torch.zeros((theta_num, self.nx * self.ny), device=self.device)
        wf_matrix = torch.zeros((theta_num, self.nx * self.ny, self.nx * self.ny), dtype=torch.cdouble, device=self.device)
        if tbc == 'x':
            for i_index, i in enumerate(theta_x):
                U = self.time_evolution_operator(self.T, 'x', vd, rotation_angle, theta_x=i, a=a, b=b, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
                eigvals, eigvecs = torch.linalg.eig(U)
                E_T = torch.log(eigvals) / (1j * self.T)
                E_T, idx = E_T.cpu().real.sort()  # Move to CPU for sorting
                idx = idx.to(self.device)  # Move indices back to GPU
                E_T = E_T.to(self.device)  # Move sorted quasienergies back to GPU
                eigvecs = eigvecs[:, idx]  # Sort eigenvectors on GPU
                eigenvalues_matrix[i_index] = E_T
                wf_matrix[i_index] = eigvecs

            if plot:
                fig, ax = plt.subplots(figsize=(12, 8))
                tick_label_fontsize = 32  # Set font size for tick labels
                label_fontsize = 34      # Set font size for the axis labels

                ax.tick_params(axis='x', labelsize=tick_label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_label_fontsize)

                theta_x_cpu = theta_x.cpu().numpy()  # Transfer theta_x to CPU and convert to NumPy for plotting
                eigenvalues_matrix_cpu = eigenvalues_matrix.cpu().numpy()  # Transfer eigenvalues_matrix to CPU and convert to NumPy for plotting
                
                ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
                ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
                
                for i in range(theta_num):
                    # Plot each point; here, we use numpy broadcasting to repeat theta_x[i] across the corresponding eigenvalues
                    ax.scatter([theta_x_cpu[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix_cpu[i], c='b', s=0.1)

                ax.set_xlabel(r'$\theta_{x}$', fontsize=label_fontsize)
                ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
                ax.set_xlim(0, 2 * np.pi)
                ax.set_ylim(-np.pi / self.T, np.pi / self.T)

                if save_path:
                    plt.tight_layout()
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')  # Save with tight bounding box.

                plt.show()
        elif tbc == 'y':
            for i_index, i in enumerate(theta_y):
                U = self.time_evolution_operator(self.T, 'y', vd, rotation_angle, theta_y=i, a=a, b=b, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
                eigvals, eigvecs = torch.linalg.eig(U)
                E_T = torch.log(eigvals) / (1j * self.T)
                E_T, idx = E_T.cpu().real.sort()  # Move to CPU for sorting
                idx = idx.to(self.device)  # Move indices back to GPU
                E_T = E_T.to(self.device)  # Move sorted quasienergies back to GPU
                eigvecs = eigvecs[:, idx]  # Sort eigenvectors on GPU
                eigenvalues_matrix[i_index] = E_T
                wf_matrix[i_index] = eigvecs

            if plot:
                fig, ax = plt.subplots(figsize=(12, 8))
                tick_label_fontsize = 32  # Set font size for tick labels
                label_fontsize = 34      # Set font size for the axis labels

                ax.tick_params(axis='x', labelsize=tick_label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_label_fontsize)

                theta_y_cpu = theta_y.cpu().numpy()  # Transfer theta_x to CPU and convert to NumPy for plotting
                eigenvalues_matrix_cpu = eigenvalues_matrix.cpu().numpy()  # Transfer eigenvalues_matrix to CPU and convert to NumPy for plotting
                
                ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
                ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
                
                for i in range(theta_num):
                    # Plot each point; here, we use numpy broadcasting to repeat theta_x[i] across the corresponding eigenvalues
                    ax.scatter([theta_y_cpu[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix_cpu[i], c='b', s=0.1)

                ax.set_xlabel(r'$\theta_{y}$', fontsize=label_fontsize)
                ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
                ax.set_xlim(0, 2 * np.pi)
                ax.set_ylim(-np.pi / self.T, np.pi / self.T)

                if save_path:
                    plt.tight_layout()
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')  # Save with tight bounding box.
                plt.show()
        return eigenvalues_matrix, wf_matrix
    
    # def time_evolution_Nperiod(self, N_times, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, delta=None, initialise=False, fully_disorder=True):
    #     '''The time evolution operator U(N_times * T) = [exp(-iH(T))]^N_times'''
    #     H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, delta, initialise, fully_disorder)
    #     H1 = self.Hamiltonian_tbc1(theta_y, tbc) + H_onsite
    #     H2 = self.Hamiltonian_tbc2(theta_x, tbc) + H_onsite
    #     H3 = self.Hamiltonian_tbc3(theta_y, tbc) + H_onsite
    #     H4 = self.Hamiltonian_tbc4(theta_x, tbc) + H_onsite
    #     H5 = H_onsite

    #     U1 = torch.matrix_exp(-1j * (self.T/5) * H1)
    #     U2 = torch.matrix_exp(-1j * (self.T/5) * H2)
    #     U3 = torch.matrix_exp(-1j * (self.T/5) * H3)
    #     U4 = torch.matrix_exp(-1j * (self.T/5) * H4)
    #     U5 = torch.matrix_exp(-1j * (self.T/5) * H5)

    #     U_period = U5 @ U4 @ U3 @ U2 @ U1

    #     U = torch.eye(U_period.shape[0], dtype=U_period.dtype, device=U_period.device)
    #     if N_times == 0:
    #         return U
    #     else:
    #         for _ in range(N_times):
    #             # print(_)
    #             U = U @ U_period
    #         return U
    
    def taylor_expansion_single(self, H, t, i):
        '''Compute a single term in the Taylor expansion'''
        return 1 / np.math.factorial(i) * (-1j * t) ** i * np.linalg.matrix_power(H, i)
    
    def taylor_expansion(self, H, t, n):
        '''Taylor expansion of exp(-iHt) using multiple threads'''
        U = np.zeros(H.shape, dtype=np.cdouble)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.taylor_expansion_single, H, t, i) for i in range(n + 1)]
            
            for future in futures:
                U += future.result()
        
        return U

    def time_evolution_1period(self, n, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):

        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, phi1, phi2, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_x, tbc) + H_onsite
        H2 = self.Hamiltonian_tbc2(theta_y, tbc) + H_onsite
        H3 = self.Hamiltonian_tbc3(theta_x, tbc) + H_onsite
        H4 = self.Hamiltonian_tbc4(theta_y, tbc) + H_onsite
        H5 = H_onsite
        H11 = H1.cpu().numpy()
        H22 = H2.cpu().numpy()
        H33 = H3.cpu().numpy()
        H44 = H4.cpu().numpy()
        H55 = H5.cpu().numpy()
        is_unitary = False
        while not is_unitary:
            U1 = self.taylor_expansion(H11, self.T / 5, n)
            U2 = self.taylor_expansion(H22, self.T / 5, n)
            U3 = self.taylor_expansion(H33, self.T / 5, n)
            U4 = self.taylor_expansion(H44, self.T / 5, n)
            U5 = self.taylor_expansion(H55, self.T / 5, n)
            U = U5 @ U4 @ U3 @ U2 @ U1
            U_dagger = np.conjugate(U).T
            product = np.dot(U_dagger, U)
            identity = np.eye(self.nx * self.ny, dtype=np.cdouble)
            is_unitary = np.allclose(product, identity, atol=1e-15)
            n += 1
        return U
    
    def normalize(self, U):
        '''Normalize matrix U to maintain unitarity'''
        U_np = U.cpu().numpy()
        Q, R = np.linalg.qr(U_np)
        U_normalized = torch.tensor(Q, dtype=U.dtype, device=U.device)
        return U_normalized
    
    def time_evolution_Nperiod(self, N_times, n, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, initialise=False, fully_disorder=True):
        '''Compute the time evolution operator for N periods using GPU'''
        U_initial = self.time_evolution_1period(n, tbc, vd, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, initialise, fully_disorder)
        U_period = torch.tensor(U_initial, dtype=torch.cdouble, device=self.device)
        U = torch.eye(U_period.shape[0], dtype=torch.cdouble, device=self.device)
        for _ in range(N_times):
            U = U @ U_period
            # Normalize to maintain unitarity every 5 steps
            if (_ + 1) % 5 == 0:
                U = self.normalize(U)
        return U
    
    def real_time_trans(self, N_times, n, initial_position, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        """The real time transmission Amplitute of evolved intial wavepacket at given intial_position (input) over N_times period"""
        '''The first intial_position is 1 instead of 0'''
        """The output should be a vector with dimension equal to the total numbers of sites"""
        U = self.time_evolution_Nperiod(N_times, n, tbc, vd, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)
        vector = torch.zeros(U.shape[1], dtype=torch.cdouble, device=self.device)  # Move vector to the same device as U
        vector[initial_position - 1] = 1.0
        # print(vector)
        Ua = U @ vector
        return Ua
    
    def transmission_prob(self, N_max, order, initial_position, energy, tbc, vd, rotation_angle, theta_x=0, theta_y=0, a=0, b=0, phi1=0, phi2=0, delta=None, fully_disorder=True):
        '''The transmission probability of the evolved intial wavepacket at given intial_position (input) over N_max period'''
        '''The first intial_position is 1 instead of 0'''
        '''The output should be a vector with dimension equal to the total numbers of sites'''
        nn = 0
        G_aa = torch.zeros(self.nx * self.ny, dtype=torch.cdouble, device=self.device)
        while nn < N_max+1:
            G = self.real_time_trans(nn, order, initial_position, tbc, vd, rotation_angle, theta_x, theta_y, a, b, phi1, phi2, delta, fully_disorder=fully_disorder)
            # print(nn)
            # Debug print statements
            print(f"nn: {nn}, energy: {energy}, T: {self.T}")
            print(f"energy type: {type(energy)}, nn type: {type(nn)}, T type: {type(self.T)}")
            complex_exponent = torch.tensor(1j * energy * nn * self.T, dtype=torch.cdouble, device=self.device)
            G_aa += G * torch.exp(complex_exponent)
            nn+= 1
        G_a = G_aa/(N_max+1)
        transmission_prob = torch.abs(G_a)**2
        return transmission_prob
    
    ## The following two functions "real_trans_prob_avg_dis" and "trans_prob_avg_disorder_realisation" only deal with the fully disordered case
    def real_trans_prob_avg_dis(self, N_dis, N_times, order, initial_pos, tbc, vd, theta_x=0, theta_y=0, delta=None):
        '''The disorder averaged real-time transmission probability of the evolved intial wavepacket at given intial_position (input) over N_times period averaging over N_dis realisation'''
        '''The first intial_position is 1 instead of 0'''
        N = 0
        avg = torch.zeros(self.nx * self.ny, device=self.device)
        while N < N_dis:
            self.H_disorder_cached = None
            print(N)
            real_amp = self.real_time_trans(N_times, order, initial_pos, tbc, vd, rotation_angle=0, theta_x=theta_x, theta_y=theta_y, a=0, b=0, delta=delta, fully_disorder=True)
            avg += torch.abs(real_amp)**2
            del real_amp
            torch.cuda.empty_cache()
            N += 1
        avg_tp = avg / N_dis
        del avg
        torch.cuda.empty_cache()
        return avg_tp

    def trans_prob_avg_disorder_realisation(self, N_dis, N_max, order, initial_position, energy, tbc, vd, delta=None):
        """The disorder averaged transmission probability of the evolved intial wavepacket at given intial_position (input) over N_max period averaging over N_dis realisation"""
        '''The first intial_position is 1 instead of 0'''
        '''The output should be a vector with dimension equal to the total numbers of sites'''
        N = 0
        avg = torch.zeros(self.nx * self.ny, device=self.device)
        while N < N_dis:
            self.H_disorder_cached = None  # Clear the cached disorder Hamiltonian
            print(N)
            trans_prob_single = self.transmission_prob(N_max, order, initial_position, energy, tbc, vd, rotation_angle=0, delta=delta, fully_disorder=True)
            avg += trans_prob_single  # Sum up the averages
            del trans_prob_single  # Free up memory of the single average once added
            torch.cuda.empty_cache()
            N += 1
        avg_tp = avg / N_dis
        del avg
        torch.cuda.empty_cache()
        return avg_tp

    ## The following two functions "real_trans_prob_avg_phase" and "trans_prob_avg_phase_realisation" only deal with the aperiodic case
    def real_trans_prob_avg_phase(self, N_phi, N_times, order, initial_pos, tbc, vd, rotation_angle = np.pi/4, theta_x=0, theta_y=0, delta=None):
        N = 0
        avg = torch.zeros(self.nx * self.ny, device=self.device)
        phi1_vals = np.random.uniform(0, 2*np.pi, N_phi)
        phi2_vals = np.random.uniform(0, 2*np.pi, N_phi)
        while N < N_phi:
            print(N)
            real_amp = self.real_time_trans(N_times, order, initial_pos, tbc, vd, rotation_angle, theta_x, theta_y, a=0, b=0, phi1=phi1_vals[N], phi2=phi2_vals[N], delta=delta, fully_disorder=False)
            avg += torch.abs(real_amp)**2
            del real_amp
            torch.cuda.empty_cache()
            N += 1
        avg_tp = avg / N_phi
        del avg
        torch.cuda.empty_cache()
        return avg_tp
    
    def trans_prob_avg_phase_realisation(self, N_phi, N_max, order, initial_pos, energy, tbc, vd, rotation_angle=np.pi/4, delta=None):
        N = 0
        avg = torch.zeros(self.nx * self.ny, device=self.device)
        phi1_vals = np.random.uniform(0, 2*np.pi, N_phi)
        phi2_vals = np.random.uniform(0, 2*np.pi, N_phi)
        while N < N_phi:
            print(N)
            trans_prob_single = self.transmission_prob(N_max, order, initial_pos, energy, tbc, vd, rotation_angle, phi1=phi1_vals[N], phi2=phi2_vals[N], delta=delta, fully_disorder=False)
            avg += trans_prob_single
            del trans_prob_single
            torch.cuda.empty_cache()
            N += 1
        avg_tp = avg / N_phi
        del avg
        torch.cuda.empty_cache()
        return avg_tp
    
    def plot_avg_trans_prob_dis(self, avg, figsize, save_path=None):
        '''Plot the average transmission probability of the evolved intial wavepacket at given intial_position (input) over N_max period averaging over N_dis realisation'''
        '''The first intial_position is 1 instead of 0'''
        '''The output should be a vector with dimension equal to the total numbers of sites'''
        fig, ax = plt.subplots(figsize=figsize)
        tick_label_fontsize = 32
        label_fontsize = 34
        trans_prob_cpu = avg.cpu().numpy().reshape(self.ny, self.nx)
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