#include <iostream>
#include <armadillo>

#include "kalman/Kalman_internal.h"


// testing
int main(int argc, char* argv[]) {

  std::cout << "Program started..." << std::endl;
  
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
  KalmanFilterInput test_kfin{
    .X = arma::randn(n, T), // observations matrix
    .A = arma::randn(m, m), // Transition matrix
    .C = arma::eye(n, m), // observation matrix
    .Q = arma::eye(m, m) * 0.1, // state covariance error matrix
    .R = arma::eye(n, n), // observation error covariance matrix
    .F_0 = arma::zeros(m), // first state
    .P_0 = arma::eye(m, m), // first state covariance
    .retLL = true};

  std::cout << "Address of test_kfin.X memory: " << test_kfin.X.memptr() << std::endl;

  bool compute_llik = true;

  std::cout << "Kalman Filter: \n";
  KalmanFilterResult resf = SKF_cpp(test_kfin);
  std::cout << "Test passed. Log-likelihood: " << resf.loglik;

  std::cout << "\n";

  KalmanSmootherInput test_ksmin = {
    .A = test_kfin.A,
    .ZTf = resf.F,
    .ZTp = resf.F_pred,
    .VTf = resf.P,
    .VTp = resf.P_pred,
    .K_last = resf.K_last,
    .C_last = resf.C_last,
    .F_0 = test_kfin.F_0,
    .P_0 = test_kfin.P_0,
    .nc_last = resf.nc_last,
  };

  std::cout << "Kalman Smoother: \n";
  KalmanSmootherResult ressm = FIS_cpp(test_ksmin);
  std::cout << "Test passed\n";

  std::cout << "Kalman Filter and Smoother: \n";
  KalmanSmootherResult reskfsm = SKFS_cpp(test_kfin);
  std::cout << "Test passed\n";


  return 0;
}

