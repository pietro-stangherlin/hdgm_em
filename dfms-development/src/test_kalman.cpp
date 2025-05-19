#include <iostream>
#include <armadillo>
#include "Kalman_internal.h"


// compile
// g++ test_kalman.cpp -o test_kalman.exe -O2 -std=c++17 -larmadillo -llapack -lblas -lgfortran -lquadmath -static-libgcc -static-libstdc++

int main() {
  // Example dimensions
  int T = 10;
  int n = 2;
  int m = 3;

  // Simulate small input matrices for testing
  arma::mat Y = arma::randn(n, T);
  arma::mat Z = arma::randn(n, m);
  arma::mat Tmat = arma::eye(m, m);
  arma::mat R = arma::eye(m, m);
  arma::mat H = arma::eye(n, n) * 0.1;
  arma::vec a1 = arma::zeros(m);
  arma::mat P1 = arma::eye(m, m);

  bool compute_llik = true;

  // Call your function
  KalmanFilterResult result = SKF_cpp(Y, Z, Tmat, R, H, a1, P1, compute_llik);

  std::cout << "Test passed. Log-likelihood: " << result.loglik;

  return 0;
}

