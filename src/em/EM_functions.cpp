#include <armadillo>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

#include "em/EM_functions.h"
#include "utils/covariances.h"
#include "optim/golden_search.h"

// Temp file


// covariance specification ----------------------------

// assuming no missing observations and no matrix permutations

/**
 * @description EM update for scale parameter of influence of state variables
 * one observation vector (eq. 4)
 * (common to each state)
 *
 * @param mY (matrix): (T x n) matrix of observed vector
 * with fixed effects predictions subtracted, NA allowed (NOT allowed temporarely)
 * (a sorting is assumed, example by spatial locations)
 * @param mZ (matrix): (T x s) matrix of smoothed state vectors
 * @param vbeta (vector) (p x 1) matrix (i.e. a vector) of fixed effects coef,
 * does NOT change with time
 * @param mXz (array) (s x s) non scaled transfer matrix (assumed constant in time
 * (the complete transfer matrix is scaled by alpha)
 * @param cPsm (array): (s x s x T) array of smoothed state variance matrices,
 *  each of those is accessed by cPsm.slice(t)
 */

double AlphaUpdate(const arma::mat & mY_fixed_res,
                   const arma::mat & mZ,
                   const arma::mat & mXz,
                   const arma::cube & cPsm){

  int T = mY_fixed_res.n_cols;

  double num = 0.0;
  double den = 0.0;

  for(int t = 0; t < T; t++){
    // NOTE: (mXz * mZ.col(t)) can be computed once and used also
    // in other updates
    num += arma::trace(mY_fixed_res.col(t) * (mXz * mZ.col(t)).t());
    den += arma::trace(mXz *
      (mZ.col(t) * mZ.col(t).t() + cPsm.slice(t)) * mXz.t());
  };

  // TO DO: add error message if den == 0
  return num / den;

}

/**
* @description compute the matrix S00
* @param smoothed_states (matrix): matrix of smoothed states
* @param smoothed_vars (array): array of smoothed sates variance matrices
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
* of the nondiffuse part of the initial state vector.
* @return (matrix) m x m
*/

arma::mat ComputeS00(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::vec & z0,
                     const arma::mat & P0){

  int T = z0.n_cols;

  arma::mat S00 = z0 * z0.t() + P0;

  // all except the last time T
  for(int t = 0; t < (T - 1); t++){
    S00 += smoothed_states.col(t) * smoothed_states.col(t).t() + smoothed_vars.slice(t);
  }

  return(S00);
}

/**
* @description compute the matrix S11
* @param smoothed_states (matrix): matrix of smoothed states
* @param smoothed_vars (array): array of smoothed sates variance matrices
* @param S00 (matrix) m x m as defined in the paper
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
* of the nondiffuse part of the initial state vector.
* @return (matrix) m x m
*/

arma::mat ComputeS11(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::mat & S00,
                     const arma::vec & z0,
                     const arma::mat & P0){
  int T = z0.n_cols;

  return(S00 - z0 * z0.t() - P0 +
         smoothed_states.col(T-1) * smoothed_states.col(T-1).t() + smoothed_vars.slice(T-1));
}

/**
* @description compute the matrix S11
* @param smoothed_states (matrix): matrix of smoothed states
* @param lagone_smoothed_covars (array): array of lag one
* smoothed states covariance matrices
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @return (matrix) m x m
*/

arma::mat ComputeS10(const arma::mat & smoothed_states,
                     const arma::cube & lagone_smoothed_covars,
                     const arma::vec & z0){

  int T = z0.n_cols;

  arma::mat S10 = smoothed_states.col(0) * z0.t() + lagone_smoothed_covars.slice(0);

  // all except the last time T
  for(int t = 1; t <= T - 1; t++){
    S10 += smoothed_states.col(t) * smoothed_states.col(t-1).t() + lagone_smoothed_covars.slice(t);
  }

  return(S10);
}


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
  return(arma::sum(arma::trace(S10)) / arma::sum(arma::trace(S00)));
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

// TO DO: Omega_t function in case there are some missing observations
// Along with Permutation matrix D definition

// here assuming NOT missing values
// NOTE: maybe it's also possible to define it just as a matrix
// considering only the elements in the diagonal
// since then the trace is taken
arma::mat OmegaSumUpdate(const arma::mat & mY_fixed_res,
                        const arma::mat & Zt,
                        const arma::mat & mXz,
                        const arma::cube & cPsmt,
                        double alpha){

  int T = mY_fixed_res.n_cols;
  int n = mY_fixed_res.n_rows;


  arma::mat Omega_sum(n, n, arma::fill::zeros);

  for(int t = 0; t < T; t++){
    Omega_sum += Omega_one_t(mY_fixed_res.col(t),
                             Zt.col(t),
                             mXz,
                             cPsmt.slice(t),
                             alpha);
  };

  return(Omega_sum);


};

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
 * @param theta Spatial decay parameter to evaluate
 * @param dist_matrix Distance matrix between spatial locations (p x p)
 * @param S00 Smoothed second moment of z_{t-1} (p x p)
 * @param S10 Smoothed cross-moment between z_t and z_{t-1} (p x p)
 * @param S11 Smoothed second moment of z_t (p x p)
 * @param g Autoregressive coefficient (scalar)
 * @param N Number of time observations (T)
 * @return double The value of the negative objective function at given theta
 */

double theta_negative_to_optim(double theta,
                         const arma::mat& dist_matrix,
                         const arma::mat& S00,
                         const arma::mat& S10,
                         const arma::mat& S11,
                         double g,
                         int N) {
  arma::mat Sigma_eta = ExpCor(dist_matrix, theta);

  double logdet_val = 0.0;
  double sign = 0.0;


  // NOTE: for small Sigma_eta one can keep the function like this
  // for big Sigma_eta it's convenient to compute a (ex. Cholesky) decomposition
  // of Sigma_eta once and use it to compute both the log determinant and the inverse
  arma::log_det(logdet_val, sign, Sigma_eta);

  arma::mat Sigma_eta_inv = arma::inv(Sigma_eta);
  arma::mat expr = S11 - g * S10 - g * S10.t() + g * g * S00;

  double trace_val = arma::trace(Sigma_eta_inv * expr);

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
 * @param theta0 Initial guess for theta (not used in optimization directly)
 * @param N Number of time points
 * @param lower lower bound for theta optimization
 * @param upper Upper bound for theta optimization
 * @return double Optimized value of theta that minimizes the objective
 */
double ThetaUpdate(const arma::mat& dist_matrix,
                   double g,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   double theta0,
                   int N,
                   double lower,
                   double upper) {

  auto obj_fun = [&](double theta) {
    return theta_negative_to_optim(theta, dist_matrix, S00, S10, S11, g, N);
  };

  // Diagnostic: Sample the objective function over the interval
  const int num_samples = 100;
  std::ofstream diag_file("src/debug/theta_diagnostics.csv");
  diag_file << "theta,value\n";
  for (int i = 0; i <= num_samples; ++i) {
    double theta = lower + i * (upper - lower) / num_samples;
    double val = obj_fun(theta);
    diag_file << theta << "," << val << "\n";
  }
  diag_file.close();

  double result = golden_search_minima(obj_fun, lower, upper); // avoid theta = 0
  return result;
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
