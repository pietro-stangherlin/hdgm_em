#include <armadillo>
#include <random>

#include "model_sim/hdgm_sim.h"

int main(int argc, char* argv[]) {

    arma::arma_rng::set_seed(12345);


    int n = 10; // number of observations
    int y_len = 5; // observation vector dimension

    double theta = 5;

    double sigmay = 0.1;
    double upsilon = 1.0;
    double gHDGM = 0.9;


    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) y_len = std::atoi(argv[2]);
    if (argc > 3) theta = std::atoi(argv[3]);
    if (argc > 4) sigmay = std::atoi(argv[4]);
    if (argc > 5) upsilon= std::atoi(argv[5]);
    if (argc > 6) gHDGM = std::atoi(argv[6]);

    arma::mat dist_matrix = arma::randu<arma::mat>(y_len, y_len);
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.t());  // symmetric
    dist_matrix.diag().zeros();

    arma::mat cor_matr = ExpCor(dist_matrix, theta);

    RHDGMResult result = RHDGM(n, y_len, cor_matr, sigmay, upsilon, gHDGM, arma::zeros<arma::vec>(y_len));

    result.y_vals.print("y values (y_len x n):");
    result.z_vals.print("z values (y_len x n+1):");

    return 0;
}
