#include <RcppArmadillo.h>
#include <stdio.h>

using namespace Rcpp;
using namespace std;

// Temp file


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

// [[Rcpp::export]]
float AlphaUpdate(arma::mat & mY_fixed_res,
                  arma::mat & mZ,
                  arma::mat & mXz,
                  arma::cube & cPsm){

  int T = mY_fixed_res.n_cols;

  float num = 0.0;
  float den = 0.0;

  for(int t = 0; t < T; t++){
    // NOTE: (mXz * mZ.col(t)) can be computed once and used also
    // in other updates
    num += arma::trace(mY_fixed_res.col(t) * (mXz * mZ.col(t)).t());
    den += arma::trace(mXz *
      (mZ.col(t) * mZ.col(t).t() + cPsm.slice(t)) * mXz.t());
  };

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

arma::mat ComputeS00(arma::mat & smoothed_states,
                     arma::cube & smoothed_vars,
                     arma::vec & z0,
                     arma::mat & P0){

  int T = z0.n_cols;

  arma::mat S00 = z0 * z0.t() + P0;

  // all except the last time T
  for(int t = 0; t < (T - 1); t++){
    S00 += smoothed_states.t() * smoothed_states.col(t).t() + smoothed_vars.slice(t);
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
arma::mat ComputeS11(arma::mat & smoothed_states,
                     arma::cube & smoothed_vars,
                     arma::mat & S00,
                     arma::vec & z0,
                     arma::mat & P0){

  int T = z0.n_cols;

  return(S00 - z0 * z0.t() - P0 +
         smoothed_states.col(T-1) * smoothed_states.col(T-1).t() + smoothed_vars.slice(T-1));
}

/**
* @description compute the matrix S11
* @param smoothed_states (matrix): matrix of smoothed states
* @param lagone_smoothed_covars (array): array of lag one
* smoothed sates covariance matrices
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @return (matrix) m x m
*/
arma::mat ComputeS10(arma::mat & smoothed_states,
                     arma::cube & lagone_smoothed_covars,
                     arma::vec & z0){

  int T = z0.n_cols;

  arma::mat S10 = smoothed_states.col(0) * z0.t() + lagone_smoothed_covars.slice(0);

  // all except the last time T
  for(int t = 1; t < T; t++){
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
float gUpdate(arma::mat & S00,
              arma::mat & S10){
  return(arma::sum(arma::trace(S10)) / arma::sum(arma::trace(S00)));
}

// NOTE: without missing data Omega_one_t = Omega_t

/**
* @description Omega_one_t definition (A.3)
*
* all vector and matrices are taken only in the non missing rows
* @param yt (matrix): (q x 1) matrix (i.e. vector) of observed vector at time t, NA allowed
* (a sorting is assumed, example by spatial locations)
* @param zt (matrix): (s x 1) matrix (i.e. vector) of smoothed state vector at time t
* @param alpha (num): scaling factor of the state vector on the observed vector
* in the HDGM model this is the upsilon parameter
* @param Xbetat (matrix) (q x p) matrix of fixed effects covariates at time t
* @param beta (matrix) (p x 1) matrix (i.e. a vector) of fixed effects coef,
* does NOT change with time
* @param inv_mXbeta_sum (matrix) (p x p) inverse of the sum of fixed effect model
* matrices cross product relative with only not NA observed vector rows:
* solve(sum(for (t in 1:T){t(Xbeta[lnmi[[t]],,]) %*% Xbeta[lnmi[[t]],,]})).
* this can be computed only once
* @param Xzt (matrix) (s x s) unscaled transfer matrix at time t
* @param Pt (matrix): (s x s) matrix of smoothed state variance at time t,
*
* @return (matrix): (vnmi x vnmi) matrix
*/
arma::mat Omega_one_t(arma::vec yt,
                        Xbetat = NULL,
                        beta,
                        zt,
                        Xzt,
                        vnmi,
                        alpha,
                        Pt){

  res <- yt - alpha * Xzt %*% as.matrix(zt)

  if(!is.null(Xbetat)){
    res <- res - Xbetat %*% beta
  }

  prod_matrix <- alpha * Xzt %*% zt

  add_matrix <- alpha^2 * Xzt %*% Pt %*% t(Xzt)

    return (res %*% t(res) +
            add_matrix)

}












