#pragma once

#include <RcppArmadillo.h>
#include <optional>
#include <type_traits>
#include <iostream>
#include <cstdlib> // For atoi

#include "../utils/symmetric_matr_vec.h"

// General EM updates --------------------------------

/**
 * @description compute the matrix S00
 * @param smoothed_states (matrix): matrix of smoothed states
 * @param smoothed_vars (array): array of smoothed sates variance matrices
 * @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
 * @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
 * of the nondiffuse part of the initial state vector.
 * @return (matrix) m x m
 */
template <typename CovStore>
arma::mat ComputeS00_core(const arma::mat & smoothed_states,
                     const CovStore & smoothed_vars,
                     const arma::vec & z0_smoothed,
                     const arma::mat & P0_smoothed){

  int T = smoothed_states.n_cols;
  int p = smoothed_states.n_rows;

  arma::mat S00 = z0_smoothed * z0_smoothed.t() + P0_smoothed;

  // all except the last time T
  for(int t = 0; t < (T - 1); t++){
    S00 += smoothed_states.col(t) * smoothed_states.col(t).t() + GetCov(smoothed_vars, t, p);
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
template <typename CovStore>
arma::mat ComputeS11_core(const arma::mat & smoothed_states,
                          CovStore & smoothed_vars,
                     const arma::mat & S00,
                     const arma::vec & z0_smoothed,
                     const arma::mat & P0_smoothed){
  int T = smoothed_states.n_cols;
  int p = smoothed_states.n_rows;

  return(S00 - z0_smoothed * z0_smoothed.t() - P0_smoothed +
         smoothed_states.col(T-1) * smoothed_states.col(T-1).t() + GetCov(smoothed_vars, T-1, p));
}

/**
 * @description compute the matrix S11
 * @param smoothed_states (matrix): matrix of smoothed states
 * @param lagone_smoothed_covars (array): array of lag one
 * smoothed states covariance matrices
 * @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
 * @return (matrix) m x m
 */

template <typename CovStore>
arma::mat ComputeS10_core(const arma::mat & smoothed_states,
                     const CovStore & lagone_smoothed_covars,
                     const arma::vec & z0_smoothed){

  int T = smoothed_states.n_cols;
  int p = smoothed_states.n_rows;

  arma::mat S10 = smoothed_states.col(0) * z0_smoothed.t() +
    smoothed_states.col(T-1) * smoothed_states.col(T-2).t() +
    GetCov(lagone_smoothed_covars, 0, p);

  // all except the last time T
  for(int t = 1; t <= T - 2; t++){
    S10 += smoothed_states.col(t) * smoothed_states.col(t-1).t() + GetCov(lagone_smoothed_covars, t, p);
  }

  return(S10);
}





// ---------------------- Structured -----------------------------//

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

template <typename CovStore>
double AlphaUpdate_core(const arma::mat & mY_fixed_res,
                   const arma::mat & mZ,
                   const arma::mat & mXz,
                   const CovStore & cPsm,
                   const arma::uvec &missing_indicator){

  int T = mY_fixed_res.n_cols;
  int p = mXz.n_rows;
  arma::uvec index_not_miss;
  arma::uvec t_index(1, arma::fill::zeros);

  arma::mat temp_state_moment;

  double num = 0.0;
  double den = 0.0;


  for(int t = 0; t < T; t++){
    temp_state_moment = mZ.col(t) * mZ.col(t).t() + GetCov(cPsm, t, p);

    if(missing_indicator[t] == 0){
      // NOTE: (mXz * mZ.col(t)) can be computed once and used also
      // in other updates
      num += arma::trace(mY_fixed_res.col(t) * (mXz * mZ.col(t)).t());
      den += arma::trace(mXz * temp_state_moment * mXz.t());

    }
    else{
      t_index[0] = t;
      index_not_miss = arma::find_finite(mY_fixed_res.col(t));

      num += arma::trace(mY_fixed_res.submat(index_not_miss, t_index) *
        (mXz.submat(index_not_miss, index_not_miss) * mZ.submat(index_not_miss, t_index)).t());
      den += arma::trace(mXz.submat(index_not_miss, index_not_miss) *
        temp_state_moment.submat(index_not_miss, index_not_miss) * mXz.submat(index_not_miss, index_not_miss).t());

    }

  };

  // TO DO: add error message if den == 0
  return num / den;

}

// TO DO: Omega_t function in case there are some missing observations
// Along with Permutation matrix D definition

// here assuming NOT missing values
// NOTE: maybe it's also possible to define it just as a matrix
// considering only the elements in the diagonal
// since then the trace is taken

template <typename CovStore>
arma::mat OmegaSumUpdate_core(const arma::mat & mY_fixed_res,
                         const arma::mat & Zt,
                         const arma::mat & mXz,
                         const CovStore & cPsmt,
                         double alpha,
                         const arma::uvec &missing_indicator,
                         const double previous_sigma2y){

  int T = mY_fixed_res.n_cols;
  int q = mY_fixed_res.n_rows;
  int p = Zt.n_rows;

  bool some_missing = false;


  arma::mat Omega_sum(q, q, arma::fill::zeros);

  for(int t = 0; t < T; t++){
      if(missing_indicator[t] == 1){
        some_missing = true;
      }
      Omega_sum += Omega_one_t(mY_fixed_res.col(t),
                               Zt.col(t),
                               mXz,
                               GetCov(cPsmt, t, p),
                               alpha,
                               some_missing,
                               previous_sigma2y);

      some_missing = false;
  };

  return(Omega_sum);


};

