#include <RcppArmadillo.h>
#include <iostream>
#include <cstdlib> // For atoi

#include "Kalman_internal.hpp"
#include "Kalman_internal_impl.hpp"
#include "../utils/symmetric_matr_vec.h"

// This code is an adaptaion from Sebastian Krantz DFMS package.
// https://cran.r-project.org/package=dfms

/* Linear Gaussian State-Space Kalman Filter and Smoother implementations.
 * With reference to the Shumway and Stoffer model here
 * - obsrvations and state covariance matrices are considered constant in time
 * - no exogenus variables are considered
 * (one can always consider the residuals as the new response)
 *
 * Model considered:
 * x_t = \Phi * x_{t-1} + w_t
 * y_t = A * x_t + v_t
 *
 * w_t \sim N_p(0, Q) iid;
 * v_t \sim N_p(0, R) iid;
 * w_t \perp v_t
*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Changes from github code: swap data matrix columns and rows definition
// so each observation is read by column (more efficient in Armadillo)
// instead than by row

// Kalman Filter
// for parameters description see Kalman_types.h


KalmanFilterResult SKF_cpp(const KalmanFilterInput& kf_inp) {
  int p = kf_inp.Phi.n_rows;
  int T = kf_inp.Y.n_cols;

  arma::cube Pp(p, p, T, arma::fill::zeros);
  arma::cube Pf(p, p, T, arma::fill::zeros);

  return SKF_core<arma::cube>(kf_inp, Pp, Pf);
};

KalmanFilterResultMat SKF_cpp_mat(const KalmanFilterInput& kf_inp) {
  int p = kf_inp.Phi.n_rows;
  int T = kf_inp.Y.n_cols;
  int sym_len = p * (p + 1) / 2;

  arma::mat Pp(sym_len, T, arma::fill::zeros);
  arma::mat Pf(sym_len, T, arma::fill::zeros);

  return SKF_core<arma::mat>(kf_inp, Pp, Pf);
};


// Kalman Smoother
KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp) {
  int p = ksm_inp.Phi.n_rows;
  int T = ksm_inp.xf.n_cols;

  arma::cube Ps(p, p, T, arma::fill::zeros);
  arma::cube Plos(p, p, T - 1, arma::fill::zeros);

  return FIS_core<arma::cube>(ksm_inp, Ps, Plos);
};

KalmanSmootherResultMat FIS_cpp_mat(const KalmanSmootherInputMat& ksm_inp) {
  int p = ksm_inp.Phi.n_rows;
  int T = ksm_inp.xf.n_cols;
  int sym_len = p * (p + 1) / 2;

  arma::mat Ps(sym_len, T, arma::fill::zeros);
  arma::mat Plos(sym_len, T - 1, arma::fill::zeros);

  return FIS_core<arma::mat>(ksm_inp, Ps, Plos);
};

// function overloading: only return type changes

// Kalman Filter and Smoother
// Only Kalman Smoother ouptput is returned
// for parameters description see Kalman_types.h
KalmanSmootherLlikResult SKFS_cpp(const KalmanFilterInput& kfsm_inp,
                                  std::type_identity<arma::cube>) {

  KalmanFilterResult kf = SKF_cpp(kfsm_inp);

  KalmanSmootherInput ksmin = {
  .Phi = kfsm_inp.Phi,
  .xf = kf.xf,
  .xp = kf.xp,
  .Pf = kf.Pf,
  .Pp = kf.Pp,
  .K_last = kf.K_last,
  .A_last = kf.A_last,
  .x_0 = kfsm_inp.x_0,
  .P_0 = kfsm_inp.P_0,
  .nc_last = kf.nc_last
  };

  KalmanSmootherResult ksmout = FIS_cpp(ksmin);


  return KalmanSmootherLlikResult(ksmout, kf.loglik);
}


KalmanSmootherLlikResultMat SKFS_cpp(const KalmanFilterInput& kfsm_inp,
                                     std::type_identity<arma::mat>) {

  KalmanFilterResultMat kf = SKF_cpp_mat(kfsm_inp);

  KalmanSmootherInputMat ksmin = {
    .Phi = kfsm_inp.Phi,
    .xf = kf.xf,
    .xp = kf.xp,
    .Pf = kf.Pf,
    .Pp = kf.Pp,
    .K_last = kf.K_last,
    .A_last = kf.A_last,
    .x_0 = kfsm_inp.x_0,
    .P_0 = kfsm_inp.P_0,
    .nc_last = kf.nc_last
  };

  KalmanSmootherResultMat ksmout = FIS_cpp_mat(ksmin);


  return KalmanSmootherLlikResultMat(ksmout, kf.loglik);
}


