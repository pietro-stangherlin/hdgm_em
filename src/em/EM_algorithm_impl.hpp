#pragma once

#include "EM_types.hpp"

#include <RcppArmadillo.h>
#include <stdio.h>
#include <limits>

#include "../kalman/Kalman_internal.hpp"
#include "EM_functions.hpp"
#include "EM_algorithm.hpp"
#include "../utils/covariances.h"
#include "../utils/symmetric_matr_vec.h"

constexpr double LOWEST_DOUBLE = std::numeric_limits<double>::lowest();

// -------------------- Unstructured Case --------------------- //
// from Parameter estimation for linear dynamical systems.
// Ghahramani, Z. & Hinton, G. E. Technical Report Technical Report CRG-TR-96-2,
// University of Totronto, Dept. of Computer Science, 1996.
template <typename CovStore>
EMOutputUnstructured UnstructuredEM_cpp_core(EMInputUnstructured& em_in){

  // ------------------ setup ------------- //
  int p = em_in.x0_in.n_rows; // state vector dimension
  int q = em_in.y.n_rows; // observation vector dimension
  int T = em_in.y.n_cols; // number of observations

  arma::mat y, Phi, A, Q, R, P0_smoothed;
  arma::vec x0_smoothed;
  double llik_prev, llik_next;
  llik_prev = LOWEST_DOUBLE;

  arma::vec diag_A;
  arma::vec diag_R;

  y = em_in.y;
  Phi = em_in.Phi_0;
  A = em_in.A_0;
  Q = em_in.Q_0;
  R = em_in.R_0;

  x0_smoothed = em_in.x0_in;
  P0_smoothed = em_in.P0_in;

  // once for all
  arma::mat sum_y_yT(q, q, arma::fill::zeros);
  for(int t = 0; t < T; t++){
    sum_y_yT += y.col(t) * y.col(t).t();
  };


  for(int iter = 1; iter < em_in.max_iter + 1; ++iter){

    std::cout << "iter" << iter << std::endl;


    ///////////////////////
    // Kalman Smoother pass
    ///////////////////////

    KalmanFilterInput kfin{
      .Y = y,
      .Phi = Phi,
      .A = A,
      .Q = Q,
      .R = R,
      .x_0 = x0_smoothed,
      .P_0 = P0_smoothed,
      .retLL = true};


    auto ksm_res = SKFS_cpp(kfin, std::type_identity<CovStore>{});

    llik_next = ksm_res.loglik;

    //DEBUG
    std::cout << "llik: " << llik_next << std::endl;

    if(llik_next < llik_prev){
      std::cout << "WARNING: Log Likelihood decreasing, returning" << std::endl;
      return EMOutputUnstructured{ .Phi = Phi, .A = A, .Q = Q, .R = R,
                                   .x0_smoothed = x0_smoothed, .P0_smoothed = P0_smoothed};

    }

    llik_prev = llik_next;

    //////////////////////////
    // EM parameters updates
    /////////////////////////

    x0_smoothed = ksm_res.x0_smoothed;
    P0_smoothed = ksm_res.P0_smoothed;

    // S matrices
    arma::mat S00 = ComputeS00_core<CovStore>(ksm_res.x_smoothed, ksm_res.P_smoothed, x0_smoothed, P0_smoothed);
    arma::mat S11 = ComputeS11_core<CovStore>(ksm_res.x_smoothed, ksm_res.P_smoothed, S00, x0_smoothed, P0_smoothed);
    arma::mat S10 = ComputeS10_core<CovStore>(ksm_res.x_smoothed, ksm_res.Lag_one_cov_smoothed, x0_smoothed);


    arma::mat sum_y_x_smooth(q, p, arma::fill::zeros);
    for(int t = 0; t < T; t++){
      sum_y_x_smooth += y.col(t) * ksm_res.x_smoothed.col(t).t();
    };

    // fix A to diagonal to ensure identificability
    A = sum_y_x_smooth * arma::inv(S11);
    diag_A = A.diag();
    A = arma::diagmat(diag_A);

    // fix R to diagonal to ensure identificability
    R = (sum_y_yT - A *  sum_y_x_smooth.t() - sum_y_x_smooth * A.t() + A * S11 * A.t()) / T ;
    diag_R = R.diag();
    R = arma::diagmat(diag_R);

    Phi = S10 * arma::inv(S00);
    Q = (S11 - Phi * S10.t()) / T;
    Q = (Q + Q.t()) * 0.5;

  }


  return EMOutputUnstructured{ .Phi = Phi, .A = A, .Q = Q, .R = R,
                               .x0_smoothed = x0_smoothed, .P0_smoothed = P0_smoothed};


};

// -------------------- Structured Case --------------------- //










