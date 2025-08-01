#pragma once

#include <RcppArmadillo.h>

struct EMInputUnstructured{
  const arma::mat& y; // observation matrix (n x T) where T = n. obs
  const arma::mat& Phi_0; // initial value transfer matrix
  const arma::mat& A_0; // initial value observation matrix
  const arma::mat& Q_0; // initial value state error covariance matrix
  const arma::mat& R_0; // initial value observation error covariance matrix
  arma::vec x0_in; // initial state
  arma::mat P0_in; // initial state covariance matrix
  int max_iter = 10; // TO change + add tolerance
  bool verbose = true;
};

struct EMInput{
  const arma::mat& y; // observation matrix (n x T) where T = n. obs
  const arma::mat& dist_matrix; // state distance matrix (complete data)
  double alpha0; // initial observation matrix scaling
  const arma::vec beta0; // initial fixed effect
  double theta0; // initial state covariance parameter (exponential)
  double g0; // initial state transition matrix scaling
  double sigma20; // initial observations variance
  const arma::vec x0_in;
  const arma::mat P0_in;
  const arma::cube &Xbeta;
  const double rel_llik_tol = 1.0e-5; // stopping criterion: relative incremente log likelihood
  const double theta_lower = 1e-05; // minimum theta value
  const double theta_upper = 20; // maximum theta value -> this can be incremented but a warning is given
  int max_iter = 10; // TO change + add tolerance
  bool is_fixed_effect = false;
  bool verbose = true;
};


struct EMOutputUnstructured{
  arma::mat Phi; // transfer matrix
  arma::mat A; // observation matrix
  arma::mat Q; // state error covariance matrix
  arma::mat R; // observation error covariance matrix
  arma::vec x0_smoothed; // smoothed first state value
  arma::mat P0_smoothed; // smoothed first state covariance value
  double llik;
  int niter;
};


struct EMOutput{
  arma::mat par_history; // matrix (k x iter) each column is iter value of (alpha,theta,g, sigma2)^T
  arma::mat beta_history; // (p x iter) each column is a beta (fixed effect value)
  double llik;
  int niter;
};
