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
    S00 += smoothed_states.col(t) * smoothed_states.col(t).t() + GetCov(smoothed_vars.Pp, t, p);
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
         smoothed_states.col(T-1) * smoothed_states.col(T-1).t() + GetCov(smoothed_vars.Pp, T-1, p));
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

  arma::mat S10 = smoothed_states.col(0) * z0_smoothed.t() + lagone_smoothed_covars.slice(0);

  // all except the last time T
  for(int t = 1; t <= T - 1; t++){
    S10 += smoothed_states.col(t) * smoothed_states.col(t-1).t() + GetCov(lagone_smoothed_covars.Pp, t, p);
  }

  return(S10);
}





// ---------------------- Structured -----------------------------//

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
                         CovStore & cPsmt,
                         double alpha){

  int T = mY_fixed_res.n_cols;
  int q = mY_fixed_res.n_rows;
  int p = Zt.n_rows;


  arma::mat Omega_sum(q, q, arma::fill::zeros);

  for(int t = 0; t < T; t++){
    Omega_sum += Omega_one_t(mY_fixed_res.col(t),
                             Zt.col(t),
                             mXz,
                             GetCov(cPsmt, t, p),
                             alpha);
  };

  return(Omega_sum);


};

