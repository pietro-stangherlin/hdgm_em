#include <RcppArmadillo.h>
#include<optional>
#pragma once

struct EMInput{
  const arma::mat& y; // observation matrix (n x T) where T = n. obs
  const arma::mat& dist_matrix; // state distance matrix (complete data)
  double alpha0; // initial observation matrix scaling
  const arma::vec beta0; // initial fixed effect
  double theta0; // initial state covariance parameter (exponential)
  double v0; // initial state scale covariance parameter (exponential)
  double g0; // initial state transition matrix scaling
  double sigma20; // initial observations variance
  std::optional<arma::cube> Xbeta = std::nullopt;
  std::optional<arma::vec> z0_in = std::nullopt;
  std::optional<arma::mat> P0_in = std::nullopt;
  const std::array<double,2> theta_v_step = {0.01, 0.01}; // step for each variable nelder-mead step
  const double var_terminating_lim = 1.0e-10; // stopping criterion for nelder-mead method: variance of values
  const int nelder_mead_max_iter = 50; // max iteration nelder-mead
  int max_iter = 10; // TO change + add tolerance
  bool verbose = true;
};


struct EMOutput{
  arma::mat par_history; // matrix (k x iter) each column is iter value of (alpha,theta,g, sigma2)^T
  arma::mat beta_history; // (p x iter) each column is a beta (fixed effect value)
};

// change return type
EMOutput EMHDGM_cpp(EMInput);
