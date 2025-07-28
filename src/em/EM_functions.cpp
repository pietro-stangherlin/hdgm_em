#include <RcppArmadillo.h>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

#include "EM_functions.hpp"
#include "../utils/symmetric_matr_vec.h"
#include "../utils/covariances.h"
#include "../optim/brent.hpp"
#include "../optim/nelder_mead.h"

// General EM updates --------------------------------

// Unstructured EM updates ----------------------------

// Structured EM updates -------------------------------

// covariance specification ----------------------------

// assuming no missing observations and no matrix permutations


// eq. (7)
// where S10 and S11 are defined in the article
// and are function of the (Kalman) smoothed vectors
/**
* @description EM update for the g constant, this is g_HDGM in the HDGM paper
* i.e. the autoregressive coefficient
* @param S00 (matrix): given the kalman smoothed states
* matrix_sum(for(t in 1:T)({z_{t-1} %*% t(z_{t-1}) + P_{t-1}}) and
* P_{t-1} is the smoothed variance at time t-1
* @param S10 (matrix): given the kalman smoothed states
* matrix_sum(for(t in 1:T)({z_{t} %*% t(z_{t-1}) + P_{t,t-1}}) and
* P_{t,t-1} is the smoothed covariance between times t and t-1
*
* @return (num)
*/

double gUpdate(const arma::mat & S00,
               const arma::mat & S10){
  return(arma::trace(S10) / arma::trace(S00));
}

// NOTE: without missing data Omega_one_t = Omega_t

// NOTE: as in alpha Update, for each iteration
// mXzt * zZt can be computed once

/**
* @description Omega_one_t definition (A.3)
*
* all vector and matrices are taken only in the non missing rows
* @param yt (matrix): (q x 1) matrix (i.e. vector) of observed vector at time t, NA allowed
* (a sorting is assumed, example by spatial locations)
* @param zt (matrix): (s x 1) matrix (i.e. vector) of smoothed state vector at time t
* @param alpha (num): scaling factor of the state vector on the observed vector
* in the HDGM model this is the upsilon parameter
* @param Xzt (matrix) (s x s) unscaled transfer matrix at time t
* @param Pt (matrix): (s x s) matrix of smoothed state variance at time t
*/

arma::mat Omega_one_t(const arma::vec & vY_fixed_res_t,
                      const arma::vec & vZt,
                      const arma::mat & mXz,
                      const arma::mat & mPsmt,
                      double alpha){

  arma::vec res = vY_fixed_res_t - alpha * mXz * vZt;

    return (res * res.t() +
            alpha * alpha * mXz * mPsmt * mXz.t());

}


double Sigma2Update(const arma::mat& Omega_sum,
                    const int n, // dimension of observation vector
                    const int T){ // number of observations

  // TO DO: take sum of traces instead of trace of sums
  return (arma::trace(Omega_sum) / (T * n));
};


/**
 * @brief Computes the negative expected complete-data log-likelihood
 *        (up to a constant) for the HDGM model, to be minimized over theta.
 *
 * @param log_theta: log of the exponential spatila correlation matrix
 * @param dist_matrix Distance matrix between spatial locations (p x p)
 * @param S00 Smoothed second moment of x_{t-1} (p x p)
 * @param S10 Smoothed cross-moment between x_t and x_{t-1} (p x p)
 * @param S11 Smoothed second moment of x_t (p x p)
 * @param Phi Autoregressive Transition Matrix
 * @param T Number of time observations
 * @return double The value of the negative objective function at given theta
 */
double LogThetaNegativeToOptim(const double log_theta,
                             const arma::mat &dist_matrix,
                             const arma::mat &S00,
                             const arma::mat &S10,
                             const arma::mat &S11,
                             const arma::mat &Phi,
                             const int &T){

  int p = S00.n_cols;

  double logdet_val = 0.0;
  double det_sign = 0.0;

  arma::mat Sigma_eta = ExpCor(dist_matrix, exp(log_theta));

  arma::log_det(logdet_val, det_sign, Sigma_eta);

  arma::mat Sigma_eta_inv = arma::inv(Sigma_eta);
  arma::mat expr = S11 - S10 * Phi.t() - Phi * S10.t() + Phi * S00 * Phi.t();

  double trace_val = arma::trace(Sigma_eta_inv * expr);

  return T * logdet_val + trace_val;

};

/**
 * @brief Performs the EM update of the spatial decay parameter theta
 *        in the Hierarchical Dynamic Gaussian Model (HDGM).
 *
 * @param dist_matrix p x p distance matrix between spatial locations
 * @param Phi Autoregressive transition matrix
 * @param S00 Smoothed second moment of x_{t-1} over time (p x p)
 * @param S10 Smoothed cross-moment of x_t and x_{t-1} over time (p x p)
 * @param S11 Smoothed second moment of x_t over time (p x p)
 * @param T Number of time points
 * @return double Optimized value of theta that minimizes the objective
 */
double ThetaUpdate(const arma::mat &dist_matrix,
                   const arma::mat &Phi,
                  const arma::mat& S00,
                  const arma::mat& S10,
                  const arma::mat& S11,
                  int &T,
                  double theta_lower,
                  double theta_upper,
                  int brent_max_iter){


  auto obj_fun = [&](const double &log_theta) {
    return LogThetaNegativeToOptim(log_theta, dist_matrix, S00, S10, S11, Phi, T);
  };


  double result = brent::brent_minimize(
    obj_fun,
    theta_lower, theta_upper, // min and max search interval
    brent_max_iter); // max iter

  // add checks if result is on the border of its parameter space

  return exp(result);
}


/**
 * @brief Computes the negative expected complete-data log-likelihood
 *        (up to a constant) for the HDGM model, to be minimized over theta.
 *
 * @param theta_v array of two elements
 * the first is theta: Spatial decay parameter to evaluate,
 * the second is sigma_z which is the state innovation covariance scaling standard
 * deviation
 * @param dist_matrix Distance matrix between spatial locations (p x p)
 * @param S00 Smoothed second moment of z_{t-1} (p x p)
 * @param S10 Smoothed cross-moment between z_t and z_{t-1} (p x p)
 * @param S11 Smoothed second moment of z_t (p x p)
 * @param g Autoregressive coefficient (scalar)
 * @param N Number of time observations (T)
 * @return double The value of the negative objective function at given theta
 */

double theta_v_negative_to_optim_log_scale(const std::array<double,2>& log_theta_v,
                         const arma::mat &dist_matrix,
                         const arma::mat &S00,
                         const arma::mat &S10,
                         const arma::mat &S11,
                         const double & g,
                         const int& N) {

  int p;
  p = S00.n_cols;

  arma::mat Sigma_eta = exp(log_theta_v[1]) * ExpCor(dist_matrix, exp(log_theta_v[0]));
  // debug
  // std::cout << "Sigma_eta" << Sigma_eta << std::endl;

  // std::cout << "inside negative to optim: " << std::endl;
  // std::cout << "theta_v[0]: " << theta_v[0] << std::endl;
  // std::cout << "theta_v[1]: " << theta_v[1] << std::endl;
  // std::cout << "dist_matrix: " << dist_matrix << std::endl;

  double logdet_val = 0.0;
  double sign = 0.0;

  // NOTE: for small Sigma_eta one can keep the function like this
  // for big Sigma_eta it's convenient to compute a (ex. Cholesky) decomposition
  // of Sigma_eta once and use it to compute both the log determinant and the inverse
  arma::log_det(logdet_val, sign, Sigma_eta);

  // debug
  //std::cout << "logdet_val" << logdet_val << std::endl;

  arma::mat G(p, p, arma::fill::eye);
  G = g * G;

  arma::mat Sigma_eta_inv = arma::inv(Sigma_eta);
  arma::mat expr = S11 - S10 * G.t() - G * S10.t() + G * S00 * G.t();

  // debug
  //std::cout << "Sigma_eta_inv" << Sigma_eta_inv << std::endl;
  //std::cout << "expr" << expr << std::endl;

  double trace_val = arma::trace(Sigma_eta_inv * expr);

  // debug
  //std::cout << "trace_val" << trace_val << std::endl;

  return N * logdet_val + trace_val;
}


// NOTE: to change if one (reasonably) wants to add a scale factor to state error covariance
// otherwise we make the unreasonable assumption of unit variance for state error.
/**
 * @brief Performs the EM update of the spatial decay parameter theta
 *        in the Hierarchical Dynamic Gaussian Model (HDGM).
 *
 * @param dist_matrix p x p distance matrix between spatial locations
 * @param g Autoregressive coefficient of the hidden state process
 * @param S00 Smoothed second moment of z_{t-1} over time (p x p)
 * @param S10 Smoothed cross-moment of z_t and z_{t-1} over time (p x p)
 * @param S11 Smoothed second moment of z_t over time (p x p)
 * @param theta_v0 Initial guess for theta_v
 * @param theta_v_step: step for each variable nelder-mead step
 * @param var_terminating_lim: stopping criterion for nelder-mead method: variance of values
 * @param max_iter: max iteration nelder-mead
 * @param N Number of time points
 * @return double Optimized value of theta that minimizes the objective
 */
std::array<double,2> ThetaVUpdate(const arma::mat& dist_matrix,
                   double& g,
                   int& N,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   const std::array<double,2>& theta_v0,
                   const std::array<double,2>& theta_v_step,
                   const double& var_terminating_lim) {

  // debug
  // std::cout << "inside ThetaVUpdate:" << std::endl;
  arma::mat Sigma_eta = theta_v0[1] * ExpCor(dist_matrix, theta_v0[0]);
  // std::cout << "theta_v[0]" << theta_v0[0] << std::endl;
  // std::cout << "theta_v[1]" << theta_v0[1] << std::endl;
  // std::cout << "dist_matrix\n" << dist_matrix << std::endl;
  // std::cout << "Sigma_eta\n" << Sigma_eta << std::endl;
  //
  // std::cout << "S00\n" << S00 << std::endl;
  // std::cout << "S10\n" << S10 << std::endl;
  // std::cout << "S11\n" << S11 << std::endl;

  auto obj_fun = [&](const std::array<double,2>& log_theta_v) {
    return theta_v_negative_to_optim_log_scale(log_theta_v, dist_matrix, S00, S10, S11, g, N);
  };

  // std::cout << "before nelder mead:" << std::endl;

  std::array<double,2> log_theta_v0 = {log(theta_v0[0]), log(theta_v0[1])};

  nelder_mead_result<double,2> result = nelder_mead<double,2>(
    obj_fun,
    theta_v0,
    var_terminating_lim, // the terminating limit for the variance of function values
    theta_v_step
  );

  // std::cout << "after nelder mead:" << std::endl;

  std::array<double,2> res = { exp(result.xmin[0]), exp(result.xmin[1]) };

  return res;
}

/**
 * @brief EM update for fixed-effect coefficients (Equation 5 in HDGM model), no missing values version.
 *
 * Estimates the fixed effects `beta` given full observation matrix `y`, smoothed state vectors `z`,
 * and covariate matrices `Xbeta`, assuming no missing data.
 *
 * @param Xbeta Array of shape (T elements of q x p), covariate matrices for each time point
 * @param y Matrix of observed vectors (q x T), no missing values assumed
 * @param z Matrix of smoothed hidden states (s x T)
 * @param alpha Scalar coefficient (upsilon parameter in HDGM)
 * @param Xz Transfer matrix from state to observation space (q x s)
 * @param inv_mXbeta_sum Precomputed inverse of ∑ₜ Xbetaᵗ Xbeta (p x p)
 *
 * @return arma::vec Estimated fixed-effect coefficients (p x 1)
 */
arma::vec BetaUpdate(const arma::cube& Xbeta,  // T elements of (q x p)
                     const arma::mat& y,                    // (q x T)
                     const arma::mat& z,                    // (s x T)
                     double alpha,
                     const arma::mat& Xz,                   // (q x s)
                     const arma::mat& inv_mXbeta_sum)       // (p x p)
{
  int p = inv_mXbeta_sum.n_rows;
  arma::vec right_term = arma::zeros(p);

  int T = y.n_cols;

  for (int t = 0; t < T; ++t) {
    right_term += Xbeta.slice(t).t() * (y.col(t) - alpha * Xz * z.col(t)); // (p)
  }

  return inv_mXbeta_sum * right_term;
}
