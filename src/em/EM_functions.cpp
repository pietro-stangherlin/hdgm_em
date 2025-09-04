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

arma::mat Omega_one_t(const arma::vec vY_fixed_res_t,
                      const arma::vec vZt,
                      const arma::mat mXz,
                      const arma::mat mPsmt,
                      double alpha,
                      const bool some_missing,
                      const double previous_sigma2y){

  int q = vY_fixed_res_t.size();
  arma::vec temp_vec;
  arma::mat res_mat(q, q, arma::fill::eye);

  // std::cout << "some_missing" << std::endl;
  // std::cout << some_missing << std::endl;

  if(some_missing == false){
    temp_vec = vY_fixed_res_t - alpha * mXz * vZt;
    res_mat = temp_vec * temp_vec.t() +
      alpha * alpha * mXz * mPsmt * mXz.t();
  }
  else{
    // used to define the permutation matrix
    arma::uvec finite_idx = arma::find_finite(vY_fixed_res_t);
    int len_finite = finite_idx.n_elem;
    arma::uvec nonfinite_idx = arma::find_nonfinite(vY_fixed_res_t);
    int len_nonfinite = nonfinite_idx.n_elem;

    // DEBUG
    // std::cout << "finite_idx" << std::endl;
    // std::cout << finite_idx << std::endl;
    //
    // std::cout << "nonfinite_idx" << std::endl;
    // std::cout << nonfinite_idx << std::endl;

    arma::uvec combined = arma::join_vert(finite_idx, nonfinite_idx);

    // std::cout << "combined" << std::endl;
    // std::cout << combined << std::endl;

    arma::mat perm_matr = MakePermutMatrix(combined);

    // compute submatrix quantities
    temp_vec = vY_fixed_res_t.elem(finite_idx) - alpha * mXz.submat(finite_idx, finite_idx) * vZt.elem(finite_idx);
    arma::mat omega_mat = temp_vec * temp_vec.t() +
      alpha * alpha * mXz.submat(finite_idx,finite_idx) *
      mPsmt.submat(finite_idx,finite_idx) * mXz.submat(finite_idx,finite_idx).t();

    arma::mat R22(len_nonfinite, len_nonfinite, arma::fill::eye);

    res_mat.submat(0, 0, len_finite - 1, len_finite - 1) = omega_mat;
    res_mat.submat(len_finite, len_finite, q - 1, q - 1) = previous_sigma2y * R22;

    res_mat = perm_matr * res_mat * perm_matr.t();
  }


    return res_mat;

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
 * @param H = S11 + S10 * Phi_temp + Phi_temp * S10.t() + Phi_temp * S00 * Phi_temp.t();
 * @param T Number of time observations
 * @return double The value of the negative objective function at given theta
 */
double LogThetaNegativeToOptim(const double log_theta,
                             const arma::mat &dist_matrix,
                             const arma::mat &H,
                             const int &T){

  double logdet_val = 0.0;
  double det_sign = 0.0;

  arma::mat Sigma_eta = ExpCor(dist_matrix, exp(log_theta));

  arma::log_det(logdet_val, det_sign, Sigma_eta);

  arma::mat Sigma_eta_inv = arma::inv(Sigma_eta);

  double trace_val = arma::trace(Sigma_eta_inv * H);

  return T * logdet_val + trace_val;

};

/**
 * @brief Performs the EM update of the spatial decay parameter theta
 *        in the Hierarchical Dynamic Gaussian Model (HDGM).
 *
 * @param dist_matrix p x p distance matrix between spatial locations
 * @param H = S11 + S10 * Phi_temp + Phi_temp * S10.t() + Phi_temp * S00 * Phi_temp.t();
 * @param T Number of time points
 * @return double Optimized value of theta that minimizes the objective
 */
double ThetaUpdate(const arma::mat &dist_matrix,
                   const arma::mat &H,
                  int &T,
                  double theta_lower,
                  double theta_upper,
                  int brent_max_iter){


  auto obj_fun = [&](const double &log_theta) {
    return LogThetaNegativeToOptim(log_theta, dist_matrix, H, T);
  };

  double log_inf = std::log(theta_lower);
  double log_sup = std::log(theta_upper);

  double result = brent::brent_minimize(
    obj_fun,
    log_inf, log_sup, // min and max search interval
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
 * @param H = S11 + S10 * Phi_temp + Phi_temp * S10.t() + Phi_temp * S00 * Phi_temp.t();
 * @param N Number of time observations (T)
 * @return double The value of the negative objective function at given theta
 */

double theta_v_negative_to_optim_log_scale(const std::array<double,2>& log_theta_v,
                         const arma::mat &dist_matrix,
                         const arma::mat &H,
                         const int& N) {
  arma::mat Sigma_eta = exp(log_theta_v[1]) * ExpCor(dist_matrix, exp(log_theta_v[0]));

  double logdet_val = 0.0;
  double sign = 0.0;

  // NOTE: for small Sigma_eta one can keep the function like this
  // for big Sigma_eta it's convenient to compute a (ex. Cholesky) decomposition
  // of Sigma_eta once and use it to compute both the log determinant and the inverse
  arma::log_det(logdet_val, sign, Sigma_eta);

  arma::mat Sigma_eta_inv = arma::inv(Sigma_eta);

  double trace_val = arma::trace(Sigma_eta_inv * H);

  return N * logdet_val + trace_val;
}


// NOTE: to change if one (reasonably) wants to add a scale factor to state error covariance
// otherwise we make the unreasonable assumption of unit variance for state error.
/**
 * @brief Performs the EM update of the spatial decay parameter theta
 *        in the Hierarchical Dynamic Gaussian Model (HDGM).
 *
 * @param dist_matrix p x p distance matrix between spatial locations
 * @param H = S11 + S10 * Phi_temp + Phi_temp * S10.t() + Phi_temp * S00 * Phi_temp.t();
 * @param theta_v0 Initial guess for theta_v
 * @param theta_v_step: step for each variable nelder-mead step
 * @param var_terminating_lim: stopping criterion for nelder-mead method: variance of values
 * @param max_iter: max iteration nelder-mead
 * @param N Number of time points
 * @return double Optimized value of theta that minimizes the objective
 */
std::array<double,2> ThetaVUpdate(const arma::mat& dist_matrix,
                                  const arma::mat &H,
                                  int& N,
                                  const std::array<double,2>& theta_v0,
                                  const std::array<double,2>& theta_v_step,
                                  const double& var_terminating_lim) {

  auto obj_fun = [&](const std::array<double,2>& log_theta_v) {
    return theta_v_negative_to_optim_log_scale(log_theta_v, dist_matrix, H, N);
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
                     const arma::mat& inv_mXbeta_sum, // (p x p)
                     const arma::uvec &missing_indicator)
{
  int p = inv_mXbeta_sum.n_rows;
  arma::vec right_term = arma::zeros(p);
  arma::uvec index_not_miss;
  arma::uvec t_index(1, arma::fill::zeros);

  arma::vec temp_res;

  int T = y.n_cols;

  for (int t = 0; t < T; ++t) {
    temp_res = alpha * Xz * z.col(t);
    if(missing_indicator[t] == 0){
    right_term += Xbeta.slice(t).t() * (y.col(t) - temp_res);} // (p)
  else{
    t_index[0] = t;
    index_not_miss = arma::find_finite(y.col(t));
    right_term += Xbeta.slice(t).rows(index_not_miss).t() *
      (y.submat(index_not_miss, t_index) - temp_res.elem(index_not_miss));
  }
  }

  return inv_mXbeta_sum * right_term;
}

/*
 * Given a vector of indexes permutations
 * return a (square) permutation matrix of the same dimension
 * supposed to bring back the elements to their original position
 * Example: perm_indexs = (2,1,0,3)
 * in order to get back to correct order (0, 1, 2, 3)
 * we need the permutation matrix D =
 * |0, 0, 1, 0|
 * |0, 1, 0, 0|
 * |1, 0, 0, 0|
 * |0, 0, 0, 1|
 * so that D * (2,1,0,3)^\top = (0, 1, 2, 3)^\top
 * where \top stands for transposition
 */
arma::mat MakePermutMatrix(const arma::uvec perm_indexes){
  int L = perm_indexes.n_elem;
  arma::mat D(L, L, arma::fill::eye);

  for(int l = 0; l < L; ++l){
    // remove 1 from diagonal
    D(l,l) = 0;
    // insert new one out of diagonal
    D(perm_indexes[l], l) = 1;
  }

  return D;


}

