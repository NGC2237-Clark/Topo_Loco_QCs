#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <vector>

using namespace std;
using namespace Eigen;

class TbFloquetTbcCuda {
public:
    TbFloquetTbcCuda(double period, double lattice_constant, double J_coe, int ny, int nx=2)
        : T(period), a(lattice_constant), J_coe(J_coe), ny(ny), nx(nx) {
        delta_AB = M_PI / (2 * T);
        H_disorder_cached = false; // Initialize H_disorder_cached as false
        device = "cuda"; // Placeholder for device (CUDA handling would be different in C++)
    }

    VectorXi latticeNumbering() {
        VectorXi numbering = VectorXi::LinSpaced(ny * nx, 0, ny * nx - 1);
        return numbering;
    }

    MatrixXcd HamiltonianTbc1(double theta_y, const string& tbc = "y") {
        MatrixXcd H1 = MatrixXcd::Zero(nx * ny, nx * ny);
        complex<double> J_coe_complex(J_coe, 0);

        if (nx % 2 == 1) { // odd nx
            for (int i = 0; i < nx * ny; ++i) {
                int a = 2 * i;
                int b = nx + 2 * i;
                if (b < nx * ny) {
                    H1(a, b) = -J_coe_complex;
                    H1(b, a) = -conj(J_coe_complex);
                }
            }
        } else { // Even nx
            MatrixXi based_pairs = MatrixXi::Zero(nx, 2);
            based_pairs(0, 0) = 0;
            based_pairs(0, 1) = nx;
            for (int j = 1; j < nx; ++j) {
                int increment = (j == nx / 2) ? 3 : 2;
                based_pairs.row(j) = based_pairs.row(j - 1) + RowVector2i(increment, increment);
            }

            for (int i = 0; i < nx; ++i) {
                int a = based_pairs(i, 0);
                int b = based_pairs(i, 1);
                while (a < nx * ny && b < nx * ny) {
                    H1(a, b) = -J_coe_complex;
                    H1(b, a) = -conj(J_coe_complex);
                    a += 2 * nx;
                    b += 2 * nx;
                }
            }
        }

        // For the twisted boundary in the y direction
        if (tbc == "y" || tbc == "xy") {
            int p = 0;
            while (1 + 2 * p < nx && ny % 2 == 0) {
                int a = 1 + nx * (ny - 1) + 2 * p;
                int b = 1 + 2 * p;
                complex<double> theta_y_complex = exp(complex<double>(0, theta_y));
                H1(a, b) = -J_coe_complex * theta_y_complex;
                H1(b, a) = -J_coe_complex * conj(theta_y_complex);
                p += 1;
            }
        }

        return H1;
    }

private:
    double T;
    double a;
    double J_coe;
    int nx;
    int ny;
    double delta_AB;
    bool H_disorder_cached;
    string device; // Placeholder for the device
};

int main() {
    double period = 1.0;
    double lattice_constant = 1.0;
    double J_coe = 1.0;
    int ny = 2;
    int nx = 2;

    TbFloquetTbcCuda model(period, lattice_constant, J_coe, ny, nx);
    double theta_y = M_PI / 4;
    MatrixXcd H1 = model.HamiltonianTbc1(theta_y);

    cout << "Hamiltonian H1:" << endl;
    cout << H1 << endl;

    return 0;
}
