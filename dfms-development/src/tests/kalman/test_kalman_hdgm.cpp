#include <iostream>
#include <armadillo>

#include "kalman/Kalman_internal.h"
#include"model_sim/hdgm_sim.h"


// testing
int main(int argc, char* argv[]) {

  std::cout << "Program started..." << std::endl;
  
  arma::arma_rng::set_seed(12345);

    int T = 10; // number of observations
    int n = 5; // observation vector dimension

    double theta = 5;

    double sigmay = 0.1;
    double upsilon = 1.0;
    double gHDGM = 0.9;


    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) n = std::atoi(argv[2]);
    if (argc > 3) theta = std::atoi(argv[3]);
    if (argc > 4) sigmay = std::atoi(argv[4]);
    if (argc > 5) upsilon= std::atoi(argv[5]);
    if (argc > 6) gHDGM = std::atoi(argv[6]);

    arma::mat dist_matrix = arma::randu<arma::mat>(n, n);
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.t());  // symmetric
    dist_matrix.diag().zeros();

    arma::mat cor_matr = ExpCor(dist_matrix, theta);

    RHDGMResult hdgm_result = RHDGM(T, n, cor_matr, sigmay, upsilon, gHDGM, arma::zeros<arma::vec>(n));

    //hdgm_result.y_vals.print("y values (y_len x n):");
    //hdgm_result.z_vals.print("z values (y_len x n+1):");


  // Simulate small input matrices for testing
  KalmanFilterInput test_kfin{
    .X = hdgm_result.y_vals, // observations matrix
    .A =  gHDGM * arma::eye(n, n), // Transition matrix
    .C = upsilon * arma::eye(n, n), // observation matrix
    .Q = cor_matr, // state covariance error matrix
    .R = sigmay * sigmay * arma::eye(n, n), // observation error covariance matrix
    .F_0 = arma::zeros(n), // first state
    .P_0 = arma::eye(n, n), // first state covariance
    .retLL = true};

  //std::cout << "Address of test_kfin.X memory: " << test_kfin.X.memptr() << std::endl;

  bool compute_llik = true;

  std::cout << "Kalman Filter: \n";
  KalmanFilterResult resf = SKF_cpp(test_kfin);
  std::cout << "Test passed. Log-likelihood: " << resf.loglik;

  //resf.F.print("Filtered States");

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

  //ressm.F_smooth.print("Smoothed States");

  std::cout << "Kalman Filter and Smoother: \n";
  KalmanSmootherResult reskfsm = SKFS_cpp(test_kfin);
  std::cout << "Test passed\n";


  return 0;
}