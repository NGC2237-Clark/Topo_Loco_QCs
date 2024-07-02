import numpy as np
import matplotlib.pyplot as plt
## Construction of the systems:
# 1. open boundary condition in y and periodic boundary in x --COMPLETED
# 2. Edge properties: U(kx, T) --COMPLETED
# 3. twisted boundary condition in x, y (the other is OBC) or both directions --COMPLETED
# 4. quasiperiodic potential --COMPLETED

## Analysis Tools
# A. Bulk properties: U(theta_x, theta_y, T) and calculate the effective Hamiltonian and get the spectrum --COMPLETED
# B. Edge properties: U(theta_x, T) and calculate the effective Hamiltonian and get the spectrum --COMPLETED
# 1. Deformation process for the bulk --> bulk invariant: winding number AND Chern number
# 2. Deformation process for the edge --> Edge winding number
# 3. Quantised Charge Pumping (Cylindrical Geometry)
# 4. Average Level Spacing Ratio 
#         PHYSICAL REVIEW B 104, L060201 (2021) for AB tiling quasicrystal
# 5. Disordered-averaged transmission probability
# 6. Inverse Participation Ratio
# 7. Critical exponent and Harris bound
# 8. Entanglement entropy (EE)
# 9. Entanglement spectrum (ES)

# Technical Issues:
# 1. Sorting eigenvalues and eigenvectors
#      a. Sorting eigenvalues in ascending order --WORKING WELL
#      b. Sorting eigenvectors based on the sorted eigenvalues --WORKING WELL
# 2. Check the unitarity of the time evolution operator
# 3. Check the Hamiltonian model for the pbc_obc --COMPLETED
# 4. Calculating the winding numbers using the triple integrals numerically
# 5. Check the branch cut of the log(np.exp())
# 6. Optimise the current code to achieve less complexity of the algorithms --> CUDA.lib
#               ----For systems with large size: maybe needed to run code on the cluster

## Questions to ask:
# 1. The energy spectra I got from the code is not the same as the one in the paper... -- COMPLETED
# 2. How to calculate the winding numbers using the triple integrals numerically
                #   B Höckendorf et al 2017 J. Phys. A: Math. Theor. 50 295301
# 3. What does L_x ∝ L_y mean in the page  8 of the AFAI paper?
# 4. Disordered-averaged transmission probability
# 5. The difference between Anderson localisation and many-body localisation --SOLVED (The nature of the difference is the lack of the interaction terms in the Anderson localisation)
# 6. the competition between the aperiodic strength and the strength of the hopping parameter J (plot)


## Ideas and things to explore:
# 1. Choose different entanglement cuts for aperiodic, fully disordered system and see the difference in the entanglement entropy compared with the periodic system without disorder.
    # Crystalline Systems:
    # Regular Order: Crystals have a regular, repeating lattice structure, leading to well-defined quantum states that extend uniformly across the material.
    # Entanglement Entropy: In crystalline systems, due to translational symmetry, the entanglement entropy typically follows an area law for entanglement cuts. 
    #                       The entanglement entropy remains relatively predictable and homogeneous across different cuts, assuming no significant surface effects or defects.

    # Aperiodic Systems (Quasicrystals):
    # Complex Order: Aperiodic systems like quasicrystals lack translational symmetry but maintain a long-range order with complex tiling patterns. The quantum states in these systems can exhibit unusual localization and spreading due to the aperiodic potential.
    # Entanglement Entropy: The entanglement entropy in quasicrystals can show anomalous scaling behavior, often dependent on the specific nature of the aperiodic order. 
    #                       This scaling can be different for different entanglement cuts, reflecting the non-uniformity in spatial correlations and quantum state extensions.

    # Fully Disordered Systems:
    # Randomness: In fully disordered systems, there is no long-range order, and the localization of quantum states (e.g., Anderson localization) plays a significant role.
    # Entanglement Entropy: The entanglement entropy in these systems can vary dramatically based on the entanglement cut. Near critical regions of a localization transition, the entropy can be higher, reflecting significant boundary effects and the influence of localized states.

# 2. Motivated by https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.93.045001, we can set the aperiodicity in hopping (off-diagonal) term rather than diagonal term (onsite energy)

# 3. Adding interaction terms (MBL)

class tb_floquet_pbc: # Tight-binding model of square lattice with Floquet driving and periodic boundary conditions
    def __init__(self, period, lattice_constant, J_coe, num_y, num_x = 2):
        self.T = period
        # self.num_cells_y = num_cells_y
        self.ny = num_y # number of sites along the y direction
        self.nx = num_x # number of sites along the x direction
        self.a = lattice_constant
        self.J_coe = J_coe # hopping matrix element
        self.delta_AB = np.pi/(2* self.T)

    
    def Hamiltonian_pbc_obc1(self, ky, pbc = 'y'):
        """The time-independent Hamiltonian H1 for t < T/5 with periodic boundary conditions in either x, y, or both x and y directions"""
        '''Specifically, for the Hamiltonian_pbc_obc1, it will only have either periodic boundary condition in the y direction and open boundary condition in the x direction
                                                                    or open boundary condition in both directions (when ky = 0)'''
        H1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        if self.nx % 2 == 1: # odd nx
            for i in range(self.nx * self.ny):  # Iterate within bounds
                a = 2 * i
                b = self.nx + 2 * i
                if b < self.nx * self.ny:
                    H1[a, b] = -self.J_coe
                    H1[b, a] = np.conjugate(-self.J_coe)
                    # print("a=", a, "b=", b)
        else:  # Even nx
            based_pairs = np.zeros((self.nx,2))
            based_pairs[0] = [0, self.nx]
            for j in range(1, self.nx):
                if j == self.nx // 2:
                    increment =3
                else:
                    increment =2
                based_pairs[j] = based_pairs[j-1] + increment

            for a, b in based_pairs:
                while a < self.nx * self.ny and b < self.nx * self.ny:
                    H1[int(a), int(b)] = -self.J_coe
                    H1[int(b), int(a)] = np.conjugate(-self.J_coe)
                    # print("a=", a, "b=", b)
                    a += 2* self.nx
                    b += 2* self.nx
        # For the periodic boundary in the y direction
        if pbc == 'y' or pbc == 'xy':
            p = 0
            while 1+2* p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[int(a), int(b)] = -self.J_coe * np.exp(1j * ky * self.a)
                H1[int(b), int(a)] = np.conjugate(-self.J_coe * np.exp(1j * ky * self.a))
                # print("ap=", a, "bp=", b)
                p += 1
        return H1

    def Hamiltonian_pbc_obc2(self, kx, pbc = 'x'):
            '''The time-independent Hamiltonian H2 for T/5 <= t < 2T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
            H2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
            # print("H2")
            n = 1
            a = -1
            b = 0
            while n <= self.ny and a < self.nx * self.ny -1 and b < self.nx * self.ny:
                # print("starting")
                a+=2
                b+=2
                if b < n * self.nx:
                    # print("pathA")
                    H2[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H2[b, a] = np.conjugate(-self.J_coe)
                else:
                    # print("pathB")
                    n += 1
                    if self.nx % 2 == 1 and n % 2 == 0:
                        a += 0
                        b += 0
                        # print("path1")
                    elif self.nx % 2 == 1 and n % 2 != 0:
                        a += 2
                        b += 2
                        # print("path2")
                    elif self.nx % 2 == 0:
                        a += 1
                        b += 1
                    if a < self.nx * self.ny -1 and b < self.nx * self.ny:
                        H2[a, b] = -self.J_coe
                        # print("a=", a, "b=", b)
                        H2[b, a] = np.conjugate(-self.J_coe)
                if self.nx == 2:
                    a += 1
                    b += 1
                if b >= self.ny * self.nx -2:
                    # print("breaking", "n=", n, "a=", a, "b=", b)
                    break
            # For the periodic boundary in the x direction
            if pbc == 'x' or pbc == 'xy':
                p = 0
                while self.nx -1 + 2 * self.nx * p < self.nx * self.ny and self.nx % 2 == 0:
                    a = self.nx - 1 + 2 * self.nx * p
                    b = 2 * self.nx * p
                    H2[a, b] = -self.J_coe * np.exp(1j * kx * self.a)
                    H2[b, a] = np.conjugate(-self.J_coe * np.exp(1j * kx * self.a))
                    # print("ap=", a, "bp=", b)
                    p += 1
            return H2


    def Hamiltonian_pbc_obc3(self, ky, pbc = 'y'):
            '''The time-independent Hamiltonian H3 for 2T/5 <= t < 3T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
            H3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
            # print("H3")
            if self.nx % 2 == 1: # odd nx
                for i in range(self.nx * self.ny):  # Iterate within bounds
                    a = 2 * i + 1
                    b = self.nx + 2 * i + 1
                    if b < self.nx * self.ny:
                        H3[a, b] = -self.J_coe
                        H3[b, a] = np.conjugate(-self.J_coe)
                        # print("a=", a, "b=", b)
            else:  # Even nx
                n = 1
                a = 1
                b = 1 + self.nx
                if b < self.nx * self.ny:
                    H3[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H3[b, a] = np.conjugate(-self.J_coe)
                while n < self.ny and a < self.nx * self.ny -1 and b < self.nx * self.ny -1:
                    # print("starting")
                    a+=2
                    b+=2
                    if a < n * self.nx:
                        # print("pathA")
                        H3[a, b] = -self.J_coe
                        #print("a=", a, "b=", b)
                        H3[b, a] = np.conjugate(-self.J_coe)
                    else:
                        # print("pathB")
                        n += 1
                        if n % 2 == 0: # even n
                            a -= 1
                            b -= 1
                        elif n % 2 != 0 and b < self.nx * self.ny -1: # odd n
                            a += 1
                            b += 1
                        else:
                            a -= 2
                            b -= 2
                        H3[a, b] = -self.J_coe
                        # print("a=", a, "b=", b)
                        H3[b, a] = np.conjugate(-self.J_coe)
            # For the periodic boundary in the y direction
            if pbc == 'y' or pbc == 'xy':
                p = 0
                while 2* p < self.nx and self.ny % 2 == 0:
                    a = self.nx * (self.ny - 1) + 2 * p
                    b = 2 * p
                    H3[int(a), int(b)] = -self.J_coe * np.exp(1j * ky * self.a)
                    H3[int(b), int(a)] = np.conjugate(-self.J_coe * np.exp(1j * ky * self.a))
                # print("ap=", a, "bp=", b)
                    p += 1
            return H3

    def Hamiltonian_pbc_obc4(self, kx, pbc = 'x'):
            '''The time-independent Hamiltonian H4 for 3T/5 <= t < 4T/5 with periodic boundary conditions in either x, y, or both x and y directions'''
            H4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
            # print("H4")
            n = 1
            a = -2
            b = -1
            while n <= self.ny and a < self.nx * self.ny -2 and b < self.nx * self.ny-2:
                # print("starting")
                a+=2
                b+=2
                if b < n * self.nx:
                    # print("pathA")
                    H4[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H4[b, a] = np.conjugate(-self.J_coe)
                else:
                    # print("pathB")
                    n += 1
                    if self.nx % 2 == 0 and self.nx != 2: # even nx
                        a += 1
                        b += 1
                        # print("path1")
                    elif self.nx % 2 == 0 and self.nx == 2 and b < self.nx * self.ny - 2: # even nx and nx = 2
                        a += 2
                        b += 2
                        n += 1
                        # print("path2")
                    elif self.nx % 2 == 1 and n % 2 == 1: # odd nx and n is odd
                        a += 0
                        b += 0
                        # print("path3")
                    elif self.nx % 2 == 1 and n % 2 == 0: # odd nx and n is even
                        a += 2
                        b += 2
                        # print("path4")
                    else:
                        n += 1
                        a += -2
                        b += -2
                    # print("error","a=", a, "b=", b, "nnn=", n)
                    H4[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H4[b, a] = np.conjugate(-self.J_coe)
                # print("a=", a, "b=", b, "nnnn=", n)
            # For the periodic boundary in the x direction
            if pbc == 'x' or pbc == 'xy':
                p = 0
                while 2 * self.nx * (1+p) - 1 < self.nx * self.ny and self.nx % 2 == 0:
                    a = 2 * self.nx * (1+p) - 1
                    b = 2 * self.nx * p + self.nx
                    H4[a, b] = -self.J_coe * np.exp(1j * kx * self.a)
                    H4[b, a] = np.conjugate(-self.J_coe * np.exp(1j * kx * self.a))
                    # print("ap=", a, "bp=", b)
                    p += 1
            return H4

    def Hamiltonian_pbc_obc_onsite(self, delta = None):
        '''The time-independent Hamiltonian H5 for 4T/5 <= t < T '''
        H_onsite = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        if delta is None:
            delta = self.delta_AB
        i = 0
        n = 0
        while i < self.nx * self.ny:
            if i == n * self.nx:
                n += 1
                delta *= -1
            H_onsite[i, i] = delta * (-1) ** (i + 1)
            i += 1
        return H_onsite

    def Hamiltonian_pbc_obc(self, t, kx, ky, delta = None, reverse = False, pbc = 'x'):
        '''The time-independent Hamiltonian H(t) with periodic boundary conditions in the x direction and open boundary conditions in the y direction'''
        if reverse == False: # Anti-clockwise
            if t < self.T/5:
                H = self.Hamiltonian_pbc_obc1(ky, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif self.T/5 <= t < 2 * self.T/5:
                H = self.Hamiltonian_pbc_obc2(kx, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                H = self.Hamiltonian_pbc_obc3(ky, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                H = self.Hamiltonian_pbc_obc4(kx, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 4 * self.T/5 <= t < self.T:
                H = self.Hamiltonian_pbc_obc_onsite(delta)
        else: # Clockwise
            if t < self.T/5:
                H = self.Hamiltonian_pbc_obc2(kx, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif self.T/5 <= t < 2 * self.T/5:
                H = self.Hamiltonian_pbc_obc1(ky, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                H = self.Hamiltonian_pbc_obc4(kx, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                H = self.Hamiltonian_pbc_obc3(ky, pbc) + self.Hamiltonian_pbc_obc_onsite(delta)
            elif 4 * self.T/5 <= t < self.T:
                H = self.Hamiltonian_pbc_obc_onsite(delta)
        return H

    def time_evolution_operator_pbc_obc(self, t, n, kx, ky, pbc, delta = None, reverse = False):
        '''The time evolution operator U(t) = exp(-iH(t)) with periodic boundary conditions in the x direction and open boundary conditions in the y direction'''
        '''The n is the order of the Taylor expansion'''
        H_onsite = self.Hamiltonian_pbc_obc_onsite(delta)
        if reverse:
            H1 = self.Hamiltonian_pbc_obc2(kx, pbc) + H_onsite
            H2 = self.Hamiltonian_pbc_obc1(ky, pbc) + H_onsite
            H3 = self.Hamiltonian_pbc_obc4(kx, pbc) + H_onsite
            H4 = self.Hamiltonian_pbc_obc3(ky, pbc) + H_onsite
            H5 = H_onsite
        else:
            H1 = self.Hamiltonian_pbc_obc1(ky, pbc) + H_onsite
            H2 = self.Hamiltonian_pbc_obc2(kx, pbc) + H_onsite
            H3 = self.Hamiltonian_pbc_obc3(ky, pbc) + H_onsite
            H4 = self.Hamiltonian_pbc_obc4(kx, pbc) + H_onsite
            H5 = H_onsite
        is_unitary = False
        while is_unitary == False:
            if t < self.T/5:
                U = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U += 1/(np.math.factorial(i)) * (-1j * t) ** i * np.linalg.matrix_power(H1, i)
            elif self.T/5 <= t < 2 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (t - self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                U = np.dot(U2, U1)

            elif 2 * self.T/5 <= t < 3 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (t - 2 * self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                U = U3 @ U2 @ U1
        
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                    U4 += 1/(np.math.factorial(i)) * (-1j * (t - 3 * self.T/5)) ** i * np.linalg.matrix_power(H4, i)
                U = U4 @ U3 @ U2 @ U1

            elif 4 * self.T/5 <= t <= self.T:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U5 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                    U4 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H4, i)
                    U5 += 1/(np.math.factorial(i)) * (-1j * (t - 4 * self.T/5)) ** i * np.linalg.matrix_power(H5, i)
                U = U5 @ U4 @ U3 @ U2 @ U1
            U_daggar = np.conjugate(U).T
            product = np.dot(U_daggar, U)
            identity = np.eye(self.nx * self.ny, dtype=complex)
            is_unitary = np.allclose(product, identity)
            # print("is_unitary", is_unitary)
            # print(n)
            n += 1
        # print("Correct order that makes sure U is unitary", n-1)
        return U

    def quasienergy_eigenstates(self, k_num, n, delta = None, reverse = False, plot=False, save_path=None, pbc = 'x'):
        '''The quasi-energy spectrum U(kx, T) for the edge properties'''
        '''The function takes inputs the number of kx array points and the order of the Taylor expansion'''
        k_x = np.linspace(0, 2*np.pi/(self.a), k_num)
        k_y = np.linspace(0, 2*np.pi/(self.a), k_num)
        if pbc == 'x':
            eigenvalues_matrix = np.zeros((k_num, self.nx * self.ny), dtype=float)
            wf_matrix = np.zeros((k_num, self.nx * self.ny, self.nx * self.ny), dtype=complex)
            for index_i, i in enumerate(k_x):
                # print(index_i)
                U = self.time_evolution_operator_pbc_obc(self.T, n, i, 0, 'x', delta, reverse)
                eigvals, eigvecs = np.linalg.eig(U)
                # print(eigvals)
                E_T = 1j * np.log(eigvals) / self.T
                # Sort the quasienergies
                idx = E_T.argsort()
                E_T = E_T[idx]
                eigvecs = eigvecs[:, idx]
                #### form the matrices of the eigenvalues and eigenvectors
                eigenvalues_matrix[index_i] = E_T.real
                wf_matrix[index_i] = eigvecs
            if plot:
                fig, ax = plt.subplots(figsize=(8, 8))
                # Set the font size for the tick labels on the axes and the axis labels
                tick_label_fontsize = 32  # Set font size for tick labels
                label_fontsize = 34  # Set font size for the axis labels

                ax.tick_params(axis='x', labelsize=tick_label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_label_fontsize)

                ## Set the x-axis ticks
                ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
                ax.set_xticklabels(['0', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
                for i in range(k_num):
                    ax.scatter([k_x[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix[i, :], color='black', s=0.1)
                ax.set_xlabel(r'$k_{x}$', fontsize=label_fontsize)
                # ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
                #ax.set_title('Energy Spectrum of Haldane Ribbon with Zigzag Termination')
                ax.set_xlim(0, 2*np.pi/self.a)
                ax.set_ylim(-np.pi/self.T, np.pi/self.T)
                if save_path:
                    plt.tight_layout()
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')  # Save with tight bounding box.
                plt.show()
        if pbc == "xy":
            eigenvalues_matrix = np.zeros((k_num, k_num, self.nx * self.ny), dtype=float)
            wf_matrix = np.zeros((k_num, k_num, self.nx * self.ny, self.nx * self.ny), dtype=complex)
            for index_i, i in enumerate(k_x):
                for index_j, j in enumerate(k_y):
                    U = self.time_evolution_operator_pbc_obc(self.T, n, i, j, 'xy', delta, reverse)
                    eigvals, eigvecs = np.linalg.eig(U)
                    E_T = 1j * np.log(eigvals) / self.T
                    # Sort the quasienergies
                    idx = E_T.argsort()
                    E_T = E_T[idx]
                    eigvecs = eigvecs[:, idx]
                    #### form the tensors of the eigenvalues and eigenvectors
                    eigenvalues_matrix[index_i, index_j] = E_T.real
                    wf_matrix[index_i, index_j] = eigvecs
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X, Y = np.meshgrid(k_x, k_y)
                # Set the font size for the tick labels on the axes and the axis labels
                tick_label_fontsize = 22
                label_fontsize = 15
                for i in range(self.nx * self.ny):
                    ax.plot_surface(X, Y, eigenvalues_matrix[:, :, i], cmap='viridis')
                    ax.set_xlabel(r'$k_{x}$', fontsize=label_fontsize)
                    ax.set_ylabel(r'$k_{y}$', fontsize=label_fontsize)
                    ax.set_zlabel('Quasienergy', fontsize=label_fontsize)
                    ax.set_zlim(-np.pi/self.T, np.pi/self.T)
                    ax.view_init(elev=2, azim=5)  # Set the view angle here
        return eigenvalues_matrix, wf_matrix

    
class tb_floquet_tbc: # Tight-binding model of square lattice with Floquet driving and twisted boundary conditions
    def __init__(self, period, lattice_constant, J_coe, ny, nx=2):
        self.T = period
        self.nx = nx  # number of x sites
        self.ny = ny  # number of y sites
        self.a = lattice_constant
        self.J_coe = J_coe # hopping matrix element
        self.H_disorder_cached = None  # Initialize the cached disorder matrix
        self.delta_AB = np.pi/(2* self.T)

    def lattice_numbering(self):
        '''Numbering the sites in the lattice'''
        site = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                site[j,i] = j*self.nx + i
        return site
    
    ## Construction of the Hamiltonian for the twisted boundary conditions

    # Benchmark the Hamiltonian_tbc1 function: nx = 6, ny = 4: Done
    #                                          nx = 5, ny = 4: Done
    #                                          nx = 5, ny = 5: Done
    #                                          nx = 2, ny = 4: Done
    #                                          nx = 2, ny = 5: Done
    def Hamiltonian_tbc1(self, theta_y, tbc = 'y'):
        """The time-independent Hamiltonian H1 for t < T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space """
        '''Specifically, for the Hamiltonian_tbc1, it will only have either twisted boundary condition in the y direction and open boundary condition in the x direction
                                                                    or open boundary condition in both directions (when theta_y = 0)'''
        H1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        if self.nx % 2 == 1: # odd nx
            for i in range(self.nx * self.ny):  # Iterate within bounds
                a = 2 * i
                b = self.nx + 2 * i
                if b < self.nx * self.ny:
                    H1[a, b] = -self.J_coe
                    H1[b, a] = np.conjugate(-self.J_coe)
                    # print("a=", a, "b=", b)
        else:  # Even nx
            based_pairs = np.zeros((self.nx,2))
            based_pairs[0] = [0, self.nx]
            for j in range(1, self.nx):
                if j == self.nx // 2:
                    increment =3
                else:
                    increment =2
                based_pairs[j] = based_pairs[j-1] + increment

            for a, b in based_pairs:
                while a < self.nx * self.ny and b < self.nx * self.ny:
                    H1[int(a), int(b)] = -self.J_coe
                    H1[int(b), int(a)] = np.conjugate(-self.J_coe)
                    # print("a=", a, "b=", b)
                    a += 2* self.nx
                    b += 2* self.nx
            
        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 1+2* p < self.nx and self.ny % 2 == 0:
                a = 1 + self.nx * (self.ny - 1) + 2 * p
                b = 1 + 2 * p
                H1[int(a), int(b)] = -self.J_coe * np.exp(1j * theta_y)
                H1[int(b), int(a)] = np.conjugate(-self.J_coe * np.exp(1j * theta_y))
                # print("ap=", a, "bp=", b)
                p += 1
        return H1
    # Benchmark the Hamiltonian_tbc2 function: nx = 6, ny = 4: Done
    #                                          nx = 5, ny = 4: Done
    #                                          nx = 5, ny = 5: Done
    #                                          nx = 2, ny = 4: Done
    #                                          nx = 2, ny = 5: Done
    def Hamiltonian_tbc2(self, theta_x, tbc = 'x'):
        '''The time-independent Hamiltonian H2 for T/5 <= t < 2T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        # print("H2")
        n = 1
        a = -1
        b = 0
        while n <= self.ny and a < self.nx * self.ny -1 and b < self.nx * self.ny:
            # print("starting")
            a+=2
            b+=2
            if b < n * self.nx:
                # print("pathA")
                H2[a, b] = -self.J_coe
                # print("a=", a, "b=", b)
                H2[b, a] = np.conjugate(-self.J_coe)
            else:
                # print("pathB")
                n += 1
                if self.nx % 2 == 1 and n % 2 == 0:
                    a += 0
                    b += 0
                    # print("path1")
                elif self.nx % 2 == 1 and n % 2 != 0:
                    a += 2
                    b += 2
                    # print("path2")
                elif self.nx % 2 == 0:
                    a += 1
                    b += 1
                if a < self.nx * self.ny -1 and b < self.nx * self.ny:
                    H2[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H2[b, a] = np.conjugate(-self.J_coe)
            if self.nx == 2:
                a += 1
                b += 1
            if b >= self.ny * self.nx -2:
                # print("breaking", "n=", n, "a=", a, "b=", b)
                break
        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while self.nx -1 + 2 * self.nx * p < self.nx * self.ny and self.nx % 2 == 0:
                a = self.nx - 1 + 2 * self.nx * p
                b = 2 * self.nx * p
                H2[a, b] = -self.J_coe * np.exp(1j * theta_x)
                H2[b, a] = np.conjugate(-self.J_coe * np.exp(1j * theta_x))
                # print("ap=", a, "bp=", b)
                p += 1
        return H2
    # Benchmark the Hamiltonian_tbc3 function: nx = 6, ny = 4: Done
    #                                          nx = 5, ny = 4: Done
    #                                          nx = 5, ny = 5: Done
    #                                          nx = 2, ny = 4: Done
    #                                          nx = 2, ny = 5: Done
    def Hamiltonian_tbc3(self, theta_y, tbc = 'y'):
        '''The time-independent Hamiltonian H3 for 2T/5 <= t < 3T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        # print("H3")
        if self.nx % 2 == 1: # odd nx
            for i in range(self.nx * self.ny):  # Iterate within bounds
                a = 2 * i + 1
                b = self.nx + 2 * i + 1
                if b < self.nx * self.ny:
                    H3[a, b] = -self.J_coe
                    H3[b, a] = np.conjugate(-self.J_coe)
                    # print("a=", a, "b=", b)
        else:  # Even nx
            n = 1
            a = 1
            b = 1 + self.nx
            if b < self.nx * self.ny:
                H3[a, b] = -self.J_coe
                # print("a=", a, "b=", b)
                H3[b, a] = np.conjugate(-self.J_coe)
            while n < self.ny and a < self.nx * self.ny -1 and b < self.nx * self.ny -1:
                # print("starting")
                a+=2
                b+=2
                if a < n * self.nx:
                    # print("pathA")
                    H3[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H3[b, a] = np.conjugate(-self.J_coe)
                else:
                    # print("pathB")
                    n += 1
                    if n % 2 == 0: # even n
                        a -= 1
                        b -= 1
                    elif n % 2 != 0 and b < self.nx * self.ny -1: # odd n
                        a += 1
                        b += 1
                    else:
                        a -= 2
                        b -= 2
                    H3[a, b] = -self.J_coe
                    # print("a=", a, "b=", b)
                    H3[b, a] = np.conjugate(-self.J_coe)
        # For the twisted boundary in the y direction
        if tbc == 'y' or tbc == 'xy':
            p = 0
            while 2* p < self.nx and self.ny % 2 == 0:
                a = self.nx * (self.ny - 1) + 2 * p
                b = 2 * p
                H3[int(a), int(b)] = -self.J_coe * np.exp(1j * theta_y)
                H3[int(b), int(a)] = np.conjugate(-self.J_coe * np.exp(1j * theta_y))
                # print("ap=", a, "bp=", b)
                p += 1
        return H3
    # Benchmark the Hamiltonian_tbc4 function: nx = 6, ny = 4: Done
    #                                          nx = 5, ny = 4: Done
    #                                          nx = 5, ny = 5: Done
    #                                          nx = 2, ny = 4: Done
    #                                          nx = 2, ny = 5: Done
    def Hamiltonian_tbc4(self, theta_x, tbc = 'x'):
        '''The time-independent Hamiltonian H4 for 3T/5 <= t < 4T/5 with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        H4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        # print("H4")
        n = 1
        a = -2
        b = -1
        # H4[a, b] = -self.J_coe
        # print("a=", a, "b=", b, "n=", n)
        # H4[b, a] = np.conjugate(-self.J_coe)
        while n <= self.ny and a < self.nx * self.ny -2 and b < self.nx * self.ny-2:
            # print("starting")
            a+=2
            b+=2
            if b < n * self.nx:
                # print("pathA")
                H4[a, b] = -self.J_coe
                # print("a=", a, "b=", b)
                H4[b, a] = np.conjugate(-self.J_coe)
            else:
                # print("pathB")
                n += 1
                if self.nx % 2 == 0 and self.nx != 2: # even nx
                    a += 1
                    b += 1
                    # print("path1")
                elif self.nx % 2 == 0 and self.nx == 2 and b < self.nx * self.ny - 2: # even nx and nx = 2
                    a += 2
                    b += 2
                    n += 1
                    # print("path2")
                elif self.nx % 2 == 1 and n % 2 == 1: # odd nx and n is odd
                    a += 0
                    b += 0
                    # print("path3")
                elif self.nx % 2 == 1 and n % 2 == 0: # odd nx and n is even
                    a += 2
                    b += 2
                    # print("path4")
                else:
                    n += 1
                    a += -2
                    b += -2
                # print("error","a=", a, "b=", b, "nnn=", n)
                H4[a, b] = -self.J_coe
                # print("a=", a, "b=", b)
                H4[b, a] = np.conjugate(-self.J_coe)
            # print("a=", a, "b=", b, "nnnn=", n)
        # For the twisted boundary in the x direction
        if tbc == 'x' or tbc == 'xy':
            p = 0
            while 2 * self.nx * (1+p) - 1 < self.nx * self.ny and self.nx % 2 == 0:
                a = 2 * self.nx * (1+p) - 1
                b = 2 * self.nx * p + self.nx
                H4[a, b] = -self.J_coe * np.exp(1j * theta_x)
                H4[b, a] = np.conjugate(-self.J_coe * np.exp(1j * theta_x))
                # print("ap=", a, "bp=", b)
                p += 1
        return H4

    def aperiodic_Honsite(self, vd, rotation_angle=np.pi/4, a=0, b=0, contourplot=False):
        '''Adding aperiodic potential to the onsite Hamiltonian'''
        '''Used Eqn (7) in the paper: PRB 100, 144202 (2019)'''
        '''The angle rotated is in radians; a and b correspond the translation in x and y directions respectively'''
        H_aperiodic = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        sites = self.lattice_numbering()
        for i in range(self.nx):
            for j in range(self.ny):
                u = i * np.cos(rotation_angle) - j * np.sin(rotation_angle)
                v = i * np.sin(rotation_angle) + j * np.cos(rotation_angle)
                phi1 = 2 * np.pi * (a* np.cos(rotation_angle) - b * np.sin(rotation_angle))
                phi2 = 2 * np.pi * (a* np.sin(rotation_angle) + b * np.cos(rotation_angle))
                # print("site=", sites[i,j])
                H_aperiodic[int(sites[j,i]), int(sites[j,i])] = np.cos(2 * np.pi * u + phi1) + np.cos(2 * np.pi * v + phi2)
        if contourplot:
            H_ape = np.zeros((self.ny, self.nx), dtype=complex)
            for i in range(self.nx):
                for j in range(self.ny):
                    u = i * np.cos(rotation_angle) - j * np.sin(rotation_angle)
                    v = i * np.sin(rotation_angle) + j * np.cos(rotation_angle)
                    phi1 = 2 * np.pi * (a* np.cos(rotation_angle) - b * np.sin(rotation_angle))
                    phi2 = 2 * np.pi * (a* np.sin(rotation_angle) + b * np.cos(rotation_angle))
                    H_ape[j,i] = np.cos(2 * np.pi * u + phi1) + np.cos(2 * np.pi * v + phi2)
            H_ap = H_ape * -vd/2
            real_H = H_ap.real
            plt.figure(figsize=(6, 6))
            # Normalize the colormap to fit the range of potential energies
            norm = plt.Normalize(np.min(real_H), np.max(real_H))
            cmap = plt.get_cmap('viridis')
            plt.imshow(real_H, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
            plt.colorbar()
            # Adding titles and labels
            plt.xlabel('X')
            plt.ylabel('Y')
            # Set axis to be in the middle of each cell
            # plt.xticks(np.arange(real_H.shape[1]), np.arange(real_H.shape[1]))
            # plt.yticks(np.arange(real_H.shape[0]), np.arange(real_H.shape[0]))
            # Show the plot
            plt.show()
        return H_aperiodic * -vd/2

    def Hamiltonian_disorder(self, vd, contourplot=False, initialise = False):
        '''The disorder Hamiltonian adding random onsite potential to the total Hamiltonian for which is uniformly distributed in the range (-vd, vd)'''
        if self.H_disorder_cached is None or initialise:
            size = self.nx * self.ny
            random_values = np.random.uniform(-1 + 1e-10, 1 - 1e-10, size)
            self.H_disorder_cached = np.zeros((size, size), dtype=float)
            np.fill_diagonal(self.H_disorder_cached, random_values)
        # print("bound=", bound)
        disorder_matrix = self.H_disorder_cached * vd
        if contourplot:
            H_dis = np.zeros((self.ny, self.nx), dtype=float)
            sites = self.lattice_numbering()
            for j in range(self.ny):
                for i in range(self.nx):
                    label = int(sites[j,i])
                    # print("label=", label)
                    H_dis[j,i] = self.H_disorder_cached[label, label]
            H_dis *= vd
            plt.figure(figsize=(6, 6))
            # Normalize the colormap to fit the range of potential energies
            norm = plt.Normalize(np.min(H_dis), np.max(H_dis))
            cmap = plt.get_cmap('viridis')
            plt.imshow(H_dis, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
            plt.colorbar()
            # Adding titles and labels
            plt.xlabel('X')
            plt.ylabel('Y')
            # Set axis to be in the middle of each cell
            # plt.xticks(np.arange(real_H.shape[1]), np.arange(real_H.shape[1]))
            # plt.yticks(np.arange(real_H.shape[0]), np.arange(real_H.shape[0]))
            plt.show()
        return disorder_matrix

    
    def Hamiltonian_onsite(self, vd, rotation_angle= np.pi/4, a=0, b=0, delta = None, initialise = False, fully_disorder = True, contourplot=False):
        '''The time-independent Hamiltonian H5 for 4T/5 <= t < T with twisted boundary conditions in either x, y, or both x and y directions in the real space'''
        # print("H5")
        H_onsite = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
        if delta is None:
            delta = self.delta_AB
        i = 0
        n = 0
        while i < self.nx * self.ny:
            if i == n * self.nx:
                n += 1
                delta *= -1
            H_onsite[i, i] = delta * (-1) ** (i + 1)  # The onsite Hamiltonian for H_clean(t)
            i += 1
        if fully_disorder:
            H_dis = self.Hamiltonian_disorder(vd, contourplot=contourplot, initialise=initialise)
        else:
            H_dis = self.aperiodic_Honsite(vd, rotation_angle, a, b, contourplot=contourplot)
        H5 = H_onsite + H_dis
        return H5

    def Hamiltonian_tbc(self, t, tbc, vd, rotation_angle, theta_x, theta_y, a=0, b=0, delta=None, initialise = False, fully_disorder = True):
        """The Hamiltonian H(t) with twisted boundary conditions in either x, y, or both x and y directions in the real space """
        '''Twisted Boundary Conditions variable: tbc
        tbc = 'x' --> twisted boundary condition in the x direction --> choose an even number of n_x
        tbc = 'y' --> twisted boundary condition in the y direction --> choose an even number of n_y
        tbc = 'xy' --> twisted boundary condition in both x and y directions --> choose an even number of n_x and n_y'''
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, delta, initialise, fully_disorder)
        if t < self.T/5:
            # The Hamiltonian H1 at the duration 0 <= t < T/5
            H = self.Hamiltonian_tbc1(theta_y, tbc) + H_onsite
        elif self.T/5 <= t < 2 * self.T/5:
            # The Hamiltonian H2 at the duration T/5 <= t < 2T/5
            H = self.Hamiltonian_tbc2(theta_x, tbc) + H_onsite
        elif 2 * self.T/5 <= t < 3 * self.T/5:
            # The Hamiltonian H3 at the duration 2T/5 <= t < 3T/5
            H = self.Hamiltonian_tbc3(theta_y, tbc) + H_onsite
        elif 3 * self.T/5 <= t < 4 * self.T/5:
            # The Hamiltonian H4 at the duration 3T/5 <= t < 4T/5
            H = self.Hamiltonian_tbc4(theta_x, tbc) + H_onsite
        elif 4 * self.T/5 <= t < self.T:
            # The Hamiltonian H5 at the duration 4T/5 <= t < T
            H = H_onsite
        return H
    
    # Function of Time evolution operator version 1
    def time_evolution_operator(self, t, tbc, vd, rotation_angle, n, theta_x=0, theta_y=0, a=0, b=0, delta=None, initialise = False, fully_disorder = True):
        '''The time evolution operator U(t) = exp(-iH(t))
        n is the order of expansion of the time evolution operator'''
        H_onsite = self.Hamiltonian_onsite(vd, rotation_angle, a, b, delta, initialise, fully_disorder)
        H1 = self.Hamiltonian_tbc1(theta_y, tbc) + H_onsite
        H2 = self.Hamiltonian_tbc2(theta_x, tbc) + H_onsite
        H3 = self.Hamiltonian_tbc3(theta_y, tbc) + H_onsite
        H4 = self.Hamiltonian_tbc4(theta_x, tbc) + H_onsite
        H5 = H_onsite
        is_unitary = False
        while is_unitary == False:
            if t < self.T/5:
                U = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U += 1/(np.math.factorial(i)) * (-1j * t) ** i * np.linalg.matrix_power(H1, i)
            elif self.T/5 <= t < 2 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (t - self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                U = np.dot(U2, U1)
            elif 2 * self.T/5 <= t < 3 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (t - 2 * self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                U = U3 @ U2 @ U1
            elif 3 * self.T/5 <= t < 4 * self.T/5:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                    U4 += 1/(np.math.factorial(i)) * (-1j * (t - 3 * self.T/5)) ** i * np.linalg.matrix_power(H4, i)
                U = U4 @ U3 @ U2 @ U1
            elif 4 * self.T/5 <= t <= self.T:
                U1 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U2 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U3 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U4 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                U5 = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=complex)
                for i in range(0,n+1):
                    U1 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H1, i)
                    U2 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H2, i)
                    U3 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H3, i)
                    U4 += 1/(np.math.factorial(i)) * (-1j * (self.T/5)) ** i * np.linalg.matrix_power(H4, i)
                    U5 += 1/(np.math.factorial(i)) * (-1j * (t - 4 * self.T/5)) ** i * np.linalg.matrix_power(H5, i)
                U = U5 @ U4 @ U3 @ U2 @ U1
            U_daggar = np.conjugate(U).T
            product = np.dot(U_daggar, U)
            identity = np.eye(self.nx * self.ny, dtype=complex)
            is_unitary = np.allclose(product, identity)
            n += 1
        # print("Correct order that makes sure U is unitary", n-1)
        return U
    
    ## Exploring the bulk properties of the system
    ## Function 1. Quasienergies and states -- COMPLETED
    ## Function 2. Effective Hamiltonian and the deformation function of the bulk evolution operator -- COMPLETED
    ## Function 3. The winding number of the quasienergy gaps
    ## Function 4. The Chern number of the bulk bands
    ## Function 5. Level spacing statistics of the bulk evolution operator --COMPLETED
    ## Function 6. The Inverse Participation Ratios

    def quasienergies_states_bulk(self, vd, n, theta_x, theta_y, a=0, b=0, rotation_angle=np.pi/4, delta=None, initialise = False, fully_disorder = True):
        '''The quasi-energy spectrum for the bulk U(theta_x, theta_y, T) properties'''
        '''The output should be the quasienergies which are the diagonal elements of the effective Hamiltonian after diagonalisation.
           This is an intermeidate step towards the 'deformed' time-periodic evolution operator '''
        U = self.time_evolution_operator(self.T, 'xy', vd, rotation_angle, n, theta_x, theta_y, a, b, delta, initialise, fully_disorder)
        # Compute the eigenvalues and eigenvectors of the time evolution operator
        eigvals, eigvecs = np.linalg.eig(U)
        # Compute the quasienergies
        E_T = 1j * np.log(eigvals) / self.T
        E_T = E_T.real
        # Sort the quasienergies
        idx = E_T.argsort()
        E_T = E_T[idx]
        eigvecs = eigvecs[:, idx]
        return E_T, eigvecs  # The E_T is a 1D numpy array of the quasienergies so far
    
    def deformed_TEO_bulk(self, t, vd, n, theta_x, theta_y, a=0, b=0, rotation_angle=np.pi/4, delta=None, initialise = False, fully_disorder = True):
        '''The "deformed" time evolution operator U_epsilon(theta_x, theta_y, t)'''
        """Step 1: The Effective Hamiltonian: obtained from the function: quasienergies_states_bulk, 
        the effective Hamiltonian is the diagonal elements of the time evolution operator"""
        E_T, _ = self.quasienergies_states_bulk(vd, n, theta_x, theta_y, a, b, rotation_angle, delta, initialise, fully_disorder)
        exp = np.exp(-1j * E_T * t)
        H_eff = np.diag(exp)
        U = self.time_evolution_operator(t, 'xy', vd, rotation_angle, n, theta_x, theta_y, a, b, delta, initialise, fully_disorder)
        U_epsilon = np.dot(U, H_eff)
        return U_epsilon
    
    def avg_level_spacing_bulk(self, vd, n, theta_x, theta_y, a=0, b=0, rotation_angle=np.pi/4, delta=None, initialise = False, fully_disorder = True):
        '''The level spacing statistics of the bulk evolution operator'''
        E_T, _ = self.quasienergies_states_bulk(vd, n, theta_x, theta_y, a, b, rotation_angle, delta, initialise, fully_disorder)
        delta = np.diff(E_T)
        level_spacing = np.zeros(len(delta)-1)
        for i in range(len(level_spacing)):
            level_spacing[i] = np.minimum(delta[i+1], delta[i])/np.maximum(delta[i+1], delta[i])
        # print(level_spacing)
        # Calculating the average level spacing ratio
        level_spacing_avg = np.mean(level_spacing)
        return level_spacing_avg
    
    def avg_level_spacing_bulk1(self, vd, n, theta_x_num, theta_y_num, a=0, b=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True):
        '''The level spacing statistics of the bulk evolution operator averaging over all theta_x and theta_y values'''
        theta_x = np.linspace(0, 2 * np.pi, theta_x_num)
        theta_y = np.linspace(0, 2 * np.pi, theta_y_num)
        avg_level_spacings = np.zeros((theta_x_num, theta_y_num), dtype=float)
        for i_index, i in enumerate(theta_x):
            for j_index, j in enumerate(theta_y):
                avg_level_spacing = self.avg_level_spacing_bulk(vd, n, i, j, a, b, rotation_angle, delta, initialise, fully_disorder)
                avg_level_spacings[i_index, j_index] = avg_level_spacing
        avg = np.mean(avg_level_spacings)
        return avg

    def plot_avg_level_spacing_bulk(self, vd_max, vd_num, n, theta_x_num, theta_y_num, a=0, b=0, rotation_angle=np.pi/4, delta=None, initialise=False, fully_disorder=True, save_path=None):
        '''Plot the average level spacing statistics of the bulk evolution operator as the function of the disorder strength vd'''
        vd = np.linspace(np.pi/(200 * self.T), vd_max, vd_num)
        avg_level_spacings = np.zeros(vd_num, dtype=float)
        x_vals = np.zeros(vd_num, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 8))
        # Set the font size for the tick labels on the axes and the axis labels
        tick_label_fontsize = 32  # Set font size for tick labels
        label_fontsize = 34  # Set font size for the axis labels

        for indexv, i in enumerate(vd):
            avg_level_spacing = self.avg_level_spacing_bulk1(i, n, theta_x_num, theta_y_num, a, b, rotation_angle, delta, initialise, fully_disorder)
            avg_level_spacings[indexv] = avg_level_spacing
            x_vals[indexv] = i * self.T
        ax.scatter(x_vals, avg_level_spacings, c='b')
        ax.set_xlabel(r'Disorder strength, $\delta V_{d}$T', fontsize=label_fontsize)
        ax.set_ylabel('Average LSR, <r>', fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        if save_path:
            plt.tight_layout()
            fig.savefig(save_path, format='pdf', bbox_inches='tight')  # Save with tight bounding box.
        plt.show()
        return None
    
    ## Exploring the edge properties of the system
    ## Function 1. Quasienergies and states -- COMPLETED
    ## Function 2. Deformed time-periodic evolution operator
    ## Function 3. Disordered-averaged transmission probability
    ## Function 4. Level spacing statistics of the edge evolution operator
    ## Function 5. The Inverse Participation Ratios
    def quasienergies_states_edge(self, vd, rotation_angle, n, theta_x_num, a=0, b=0, delta=None, initialise=False, fully_disorder=True, plot=False, save_path=None):
        '''The quasi-energy spectrum for the edge  U(kx, T) properties'''
        '''The output should be the quasienergies which are the diagonal elements of the effective Hamiltonian after diagonalisation.
           This is an intermeidate step towards the 'deformed' time-periodic evolution operator '''
        theta_x = np.linspace(0, 2 * np.pi, theta_x_num)
        eigenvalues_matrix = np.zeros((theta_x_num, self.nx * self.ny), dtype=float)
        wf_matrix = np.zeros((theta_x_num, self.nx * self.ny, self.nx * self.ny), dtype=complex)
        for i_index, i in enumerate(theta_x):
            U = self.time_evolution_operator(self.T, 'x', vd, rotation_angle, n, i, theta_y=0, a=a, b=b, delta=delta, initialise=initialise, fully_disorder=fully_disorder)
            # Compute the eigenvalues and eigenvectors of the time evolution operator
            eigvals, eigvecs = np.linalg.eig(U)
            # Compute the quasienergies
            E_T = 1j * np.log(eigvals) / self.T
            # Sort the quasienergies
            idx = E_T.argsort()
            E_T = E_T[idx]
            eigvecs = eigvecs[:, idx]
            eigenvalues_matrix[i_index] = E_T.real
            wf_matrix[i_index] = eigvecs
        # print(theta_x)
        if plot:
            fig, ax = plt.subplots(figsize=(12,8))
            # Set the font size for the tick labels on the axes and the axis labels
            tick_label_fontsize = 32  # Set font size for tick labels
            label_fontsize = 34  # Set font size for the axis labels
            ax.tick_params(axis='x', labelsize=tick_label_fontsize)
            ax.tick_params(axis='y', labelsize=tick_label_fontsize)
            for i in range(theta_x_num):
                ax.scatter([theta_x[i]] * eigenvalues_matrix.shape[1], eigenvalues_matrix[i], c='b', s=0.1)
            ax.set_xlabel(r'$\theta_{x}$', fontsize=label_fontsize)
            ax.set_ylabel('Quasienergy', fontsize=label_fontsize)
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(-np.pi/self.T, np.pi/self.T)
            # if save_path:
            #     plt.tight_layout()
            #     fig.savefig(save_path, format='pdf', bbox_inches='tight')  # Save with tight bounding box.
            plt.show()
        return eigenvalues_matrix, wf_matrix


    # Functions of Time evolution operator version 2
    def calculate_time_evolution_segment(self, H, t_segment, n):
        U_segment = np.eye(H.shape[0], dtype=complex)  # Start with the identity matrix
        H_power = np.eye(H.shape[0], dtype=complex)  # To accumulate powers of H
        factorial = 1  # To accumulate values of factorial
        for i in range(1, n + 1):
            H_power = np.dot(H_power, H)  # Increment power of H
            factorial *= i  # Increment factorial
            U_segment += 1 / factorial * (-1j * t_segment) ** i * H_power
        return U_segment

    def time_evolution_operator2(self, t, tbc, vd, rotation_angle, n, theta_x=0, theta_y=0, a=0, b=0, delta=None):
        '''The time evolution operator U(t) = exp(-iH(t)t)'''
        # Initialize the identity matrix for the full evolution
        U = np.eye(self.nx * self.ny, dtype=complex)
        H_segments = []
        t_intervals = [0, self.T / 5, 2 * self.T / 5, 3 * self.T / 5, 4 * self.T / 5, t]
        print("t_intervals=",t_intervals)
        # Compute each segment of the Hamiltonian
        for start, end in zip(t_intervals[:-1], t_intervals[1:]):
            H = self.Hamiltonian_tbc(start, tbc, vd, rotation_angle, theta_x, theta_y, a, b, delta)
            print("t_segment=", end - start)
            H_segments.append((H, end - start))
        # Calculate the time evolution operator for each segment
        for H, t_segment in H_segments:
            U_segment = self.calculate_time_evolution_segment(H, t_segment, n)
            U = np.dot(U_segment, U)
        return U