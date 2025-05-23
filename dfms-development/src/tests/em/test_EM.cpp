#include <iostream>
#include <armadillo>
#include <optional>

#include "em/EM_algorithm.h"


// Compile
// g++ test_EM.cpp EM_algorithm.cpp EM_functions.cpp Kalman_internal.cpp -o test_em.exe -O2 -std=c++17 -larmadillo -llapack -lblas -lgfortran -lquadmath -static-libgcc -static-libstdc++


int main() {
  int n = 3;   // number of observed variables
  int T = 5;   // number of time steps
  int p = 2;   // number of fixed effects

  // Randomly generated test matrices
  arma::mat y(n, T, arma::fill::randn);               // observations
  arma::mat dist_matrix = arma::randu<arma::mat>(n, n); // distance matrix
  dist_matrix = 0.5 * (dist_matrix + dist_matrix.t());  // make symmetric
  dist_matrix.diag().zeros();                           // zero diagonal

  arma::vec beta0(p, arma::fill::randu);               // initial beta
  arma::cube Xbeta(n, p, T, arma::fill::randn);        // design matrix for fixed effects
  arma::vec z0(n, arma::fill::ones);                   // initial state mean
  arma::mat P0 = arma::eye(n, n);                      // initial state covariance

  // Construct EMInput
  EMInput em_in{
    .y = y,
    .dist_matrix = dist_matrix,
    .alpha0 = 1.0,
    .beta0 = beta0,
    .theta0 = 0.5,
    .g0 = 0.9,
    .sigma20 = 0.1,
    .Xbeta = Xbeta,
    .z0_in = z0,
    .P0_in = P0,
    .max_iter = 10,
    .sigma2_lower = 1e-5,
    .sigma2_upper = 10.0,
    .verbose = true
  };

  // Call the EM algorithm
  int result = EMHDGM(em_in);

  std::cout << "\nEMHDGM finished with return code: " << result << std::endl;
  return 0;
}

