#include <RcppArmadillo.h>
#include <iostream>
#include <optional>

#include "em/EM_algorithm.h"
#include"model_sim/hdgm_sim.h"
#include "utils/covariances.h"

int main(int argc, char* argv[]) {


  int T = 10000;   // number of time steps
  int n = 5;   // number of observed variables and states
  int p = 10;   // number of fixed effects
  int max_iter = 10; // EM max iter


    arma::arma_rng::set_seed(12345);

    double theta = 5;
    double upsilon = 3;
    double gHDGM = 0.8;
    double sigmay = 0.1;


    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) n = std::atoi(argv[2]);
    if (argc > 3) p = std::atoi(argv[3]);
    if (argc > 4) max_iter = std::atoi(argv[4]);
    if (argc > 5) theta = std::atoi(argv[5]);
    if (argc > 6) sigmay = std::atoi(argv[6]);
    if (argc > 7) upsilon= std::atoi(argv[7]); // alpha
    if (argc > 8) gHDGM = std::atoi(argv[8]);

    arma::mat dist_matrix = arma::randu<arma::mat>(n, n);
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.t());  // symmetric
    dist_matrix.diag().zeros();

    arma::mat cor_matr = ExpCor(dist_matrix, theta);

    RHDGMResult hdgm_result = RHDGM(T, n, cor_matr, sigmay, upsilon, gHDGM, arma::zeros<arma::vec>(n));

  arma::vec beta0(p, arma::fill::zeros);               // initial beta
  arma::cube Xbeta(n, p, T, arma::fill::randn);        // design matrix for fixed effects
  arma::vec z0(n, arma::fill::ones);                   // initial state mean
  arma::mat P0 = arma::eye(n, n);                      // initial state covariance

  // Construct EMInput
  EMInput em_in{
    .y = hdgm_result.y_vals,
    .dist_matrix = dist_matrix,
    .alpha0 = upsilon,
    .beta0 = beta0,
    .theta0 = theta,
    .g0 = gHDGM,
    .sigma20 = sigmay * sigmay,
    .Xbeta = Xbeta,
    .z0_in = z0,
    .P0_in = P0,
    .max_iter = max_iter,
    .theta_lower = 1e-5,
    .theta_upper = 10.0,
    .verbose = true
  };

  // Call the EM algorithm
  EMOutput result = EMHDGM(em_in);

  std::cout << "\nEMHDGM finished" << std::endl;
  std::cout << "Parameters iter history" << std::endl;
  result.par_history.print();
  std::cout << std::endl;
  std::cout << "Beta iter history" << std::endl;
  result.beta_history.print();

  return 0;
}
