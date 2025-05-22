#include <iostream>
#include <armadillo>
#include "Kalman_internal.h"


// compile
// g++ test_kalman_debug.cpp Kalman_internal.cpp -o test_kalman_debug.exe -O2 -std=c++17 -larmadillo -llapack -lblas -lgfortran -lquadmath -static-libgcc -static-libstdc++

// testing
int main(int argc, char* argv[]) {
  // Default values
  int T = 100;
  int n = 5;
  int m = 2;

  // If arguments are provided, override the defaults
  if (argc > 1) T = std::atoi(argv[1]);
  if (argc > 2) n = std::atoi(argv[2]);
  if (argc > 3) m = std::atoi(argv[3]);

  std::cout << "T = " << T << ", n = " << n << ", m = " << m << std::endl;
  // Simulate small input matrices for testing
  arma::mat X = arma::randn(n, T); // observations matrix
  arma::mat A = arma::randn(m, m); // Transition matrix
  arma::mat C = arma::eye(n, m); // observation matrix
  arma::mat Q = arma::eye(m, m) * 0.1; // state covariance error matrix
  arma::mat R = arma::eye(n, n); // observation error covariance matrix
  arma::vec F_0 = arma::zeros(m); // first state
  arma::mat P_0 = arma::eye(m, m); // first state covariance

  bool compute_llik = true;

  std::cout << "Kalman Filter: \n";
  KalmanFilterResult resf = SKFExplicitInput_cpp(X, // observation matrix: each observation is a column
                                          A, // state transition matrix
                                          C, // observation matrix
                                          Q, // state covariance error matrix
                                          R, // observation error covariance matrix
                                          F_0, // initial state
                                          P_0, // initial state covariance
                                          compute_llik);

  std::cout << "Test passed. Log-likelihood: " << resf.loglik;

  std::cout << "\n";




  return 0;
}
