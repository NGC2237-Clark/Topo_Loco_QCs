import torch
import matplotlib.pyplot as plt
import numpy as np

# Create a test class with both methods
class TestClass:
    def log_with_branchcut1(self, z, epsilonT):
        chi = torch.angle(z)
        chi = chi % (2 * torch.pi)
        chi_adjusted = torch.where(
            chi < epsilonT,
            chi,
            chi - 2 * torch.pi
        )
        return chi_adjusted * 1j

    def log_with_branch_cut(self, z, branch_cut_angle=0):
        if not isinstance(branch_cut_angle, torch.Tensor):
            branch_cut_angle = torch.tensor([branch_cut_angle], dtype=torch.float64, device=z.device)
        else:
            branch_cut_angle = branch_cut_angle.to(device=z.device, dtype=torch.float64)
        
        magnitude = torch.abs(z)
        initial_phase = torch.angle(z)
        
        branch_cut_angle = (branch_cut_angle + torch.pi) % (2 * torch.pi) - torch.pi

        adjusted_phase = initial_phase.unsqueeze(-1) - branch_cut_angle
        adjusted_phase = (adjusted_phase + torch.pi) % (2 * torch.pi) - torch.pi
        adjusted_phase += branch_cut_angle

        adjusted_phase += torch.where(adjusted_phase < branch_cut_angle, 2 * torch.pi, 0)
        adjusted_phase -= torch.where(adjusted_phase >= branch_cut_angle + 2 * torch.pi, 2 * torch.pi, 0)

        log_z = torch.log(magnitude).unsqueeze(-1) + 1j * adjusted_phase
        
        return log_z

# Create test points
theta = torch.linspace(0, 2*torch.pi, 100, dtype=torch.float64)
z = torch.exp(1j * theta)  # Points on the unit circle

# Initialize class
test = TestClass()

# Test both functions with epsilonT = 0 and epsilonT = π
epsilonT_values = [0, torch.pi]

plt.figure(figsize=(15, 10))

for i, epsilonT in enumerate(epsilonT_values):
    result1 = test.log_with_branchcut1(z, epsilonT)
    result2 = test.log_with_branch_cut(z, epsilonT)
    
    plt.subplot(2, 2, 2*i + 1)
    plt.plot(theta.numpy(), result1.imag.numpy(), label='log_with_branchcut1')
    plt.plot(theta.numpy(), result2.squeeze().imag.numpy(), '--', label='log_with_branch_cut')
    plt.title(f'Imaginary part (ε = {epsilonT})')
    plt.xlabel('θ')
    plt.ylabel('Im(log(z))')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2*i + 2)
    plt.plot(theta.numpy(), result1.real.numpy(), label='log_with_branchcut1')
    plt.plot(theta.numpy(), result2.squeeze().real.numpy(), '--', label='log_with_branch_cut')
    plt.title(f'Real part (ε = {epsilonT})')
    plt.xlabel('θ')
    plt.ylabel('Re(log(z))')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Print some specific values for comparison
print("\nComparison at specific points:")
test_indices = [0, 25, 50, 75]
for idx in test_indices:
    print(f"\nAt θ = {theta[idx]:.2f}:")
    for epsilonT in epsilonT_values:
        print(f"\nFor ε = {epsilonT}:")
        result1 = test.log_with_branchcut1(z[idx], epsilonT)
        result2 = test.log_with_branch_cut(z[idx], epsilonT)
        print(f"log_with_branchcut1: {result1}")
        print(f"log_with_branch_cut: {result2.squeeze()}")