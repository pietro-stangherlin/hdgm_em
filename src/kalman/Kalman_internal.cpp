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

  // std::cout << "Inside SKF_cpp" << std::endl;

  const int q = kf_inp.Y.n_rows; // observation vector dimensions
  const int p = kf_inp.Phi.n_rows; // state vector dimensions
  const int T = kf_inp.Y.n_cols; // number of observations
  int n_c; // used to count number of missing observations at each time

  // Allocate matrices and arrays where to store results

  // Predicted state mean and covariance
  arma::mat xp_vals(p, T, arma::fill::zeros);
  arma::cube Pxp_vals(p, p, T, arma::fill::zeros);

  // Filtered state mean and covariance
  arma::mat xf_vals(p, T, arma::fill::zeros);
  arma::cube Pxf_vals(p, p, T, arma::fill::zeros);

  double loglik = kf_inp.retLL ? 0.0 : std::numeric_limits<double>::quiet_NaN();

  double dn = 0.0;
  double detS = 0.0;

  //std::cout << "AFTER llik;\n";

  // xpt: temp predicted state
  // xft: temp filtered state

  arma::vec xpt, xft, et, yt;
  xft = kf_inp.x_0;

  // Ppt: temp predicted state covariance
  // Pft: temp filtered state covariance
  // K: temp kalman gain
  // S and VCt: temp auxiliary quantities

  arma::mat K, Ppt, Pft, S, VCt;
  Pft = kf_inp.P_0;

  // Handling missing values in the filter
  arma::mat At, Rt;
  arma::uvec nmiss, arow = arma::find_finite(kf_inp.Phi.row(0));
  if (arow.n_elem == 0) {
    throw std::runtime_error("Missing first row of transition matrix\n");
  }


  //std::cout << "[DEBUG] Before for loop" << std::endl;
  for (int t = 0; t < T; ++t) {

    // Run a prediction

    xpt = kf_inp.Phi * xft;
    Ppt = kf_inp.Phi * Pft * kf_inp.Phi.t() + kf_inp.Q;
    Ppt += Ppt.t(); // Ensure symmetry
    Ppt *= 0.5;

    // If missing observations are present at some timepoints, exclude the
    // appropriate matrix slices from the filtering procedure.
    yt = kf_inp.Y.col(t);
    nmiss = arma::find_finite(yt);
    n_c = nmiss.n_elem;
    if(n_c > 0) {
      if(n_c == q) {
        At = kf_inp.A;
        Rt = kf_inp.R;
      } else {
        At = kf_inp.A.submat(nmiss, arow);
        Rt = kf_inp.R.submat(nmiss, nmiss);
        yt = yt.elem(nmiss);
      }

      // Intermediate results
      VCt = Ppt * At.t();
      S = arma::inv_sympd(At * VCt + Rt);


      // Prediction error
      et = yt - At * xpt;
      // Kalman gain
      K = VCt * S;
      // Updated state estimate
      xft = xpt + K * et;
      // Updated state covariance estimate
      Pft = Ppt - K * At * Ppt;
      Pft += Pft.t(); // Ensure symmetry
      Pft *= 0.5;

      // Compute likelihood. Skip this part if S is not positive definite.
      if(kf_inp.retLL) {
        //TO DO: Maybe convenient to compute one decomposition
        // once and then do inverse and det
        double log_det_val;
        double det_sign;
        arma::log_det(log_det_val, det_sign, S);
        if(det_sign > 0) loglik += log_det_val - arma::as_scalar(et.t() * S * et) - dn;

      }

    }

    else { // If all missing: just prediction.
      xft = xpt;
      Pft = Ppt;
    }


    // Store predicted and filtered data needed for smoothing
    xp_vals.col(t) = xpt;
    Pxp_vals.slice(t) = Ppt;
    xf_vals.col(t) = xft;
    Pxf_vals.slice(t) = Pft;

  }


  if(kf_inp.retLL) loglik *= 0.5;

  return KalmanFilterResult{
    .xf = xf_vals, // filtered states
    .Pf = Pxf_vals, // filtered states Variances
    .xp = xp_vals, // predicted states
    .Pp = Pxp_vals, // predicted states variances
    .K_last = K, // Kalman gain last observation
    .A_last = At, // Observation matrix submatrix for non-missing (used in the smoother first step)
    .nc_last = n_c,
    .loglik = loglik
  };
}

KalmanFilterResultMat SKF_cpp_mat(const KalmanFilterInput& kf_inp) {
  int p = kf_inp.Phi.n_rows;
  int T = kf_inp.Y.n_cols;
  int sym_len = p * (p + 1) / 2;

  arma::mat Pp(sym_len, T, arma::fill::zeros);
  arma::mat Pf(sym_len, T, arma::fill::zeros);

  return SKF_core<arma::mat>(kf_inp, Pp, Pf);
};


// Kalman Smoother
// for parameters description see Kalman_types.h
KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp) {

  // std::cout << "Inside FIS_cpp" << std::endl;

  const int T = ksm_inp.xf.n_cols;
  const int p = ksm_inp.Phi.n_rows;

  arma::mat Pf = ksm_inp.Pf.slice(T-1);
  arma::mat Pp = ksm_inp.Pp.slice(T-1);
  arma::mat K_times_A; // Kalman gain last observation times observation matrix

  // allocate smoothed quantities
  arma::mat xs_vals(p, T, arma::fill::zeros);
  arma::cube Ps(p, p, T, arma::fill::zeros);
  arma::cube Plos(p, p, T, arma::fill::zeros);


  // populate last smoothed values
  xs_vals.col(T-1) = ksm_inp.xf.col(T-1); // last smoothed state = filtered state
  Ps.slice(T-1) = ksm_inp.Pf.slice(T-1); // last smoothed state cov = filtered state cov


  arma::mat Ji, Jim_tr;
  arma::mat Phi_tr = ksm_inp.Phi.t();

  K_times_A = (ksm_inp.nc_last == 0) ? arma::mat(p, p, arma::fill::zeros) : ksm_inp.K_last * ksm_inp.A_last;

  Plos.slice(T-1) = (arma::eye(p,p) - K_times_A) * ksm_inp.Phi * ksm_inp.Pf.slice(T-2);

  // Smoothed state variable and covariance
  for (int t = T - 2; t >= 0; --t) {
    arma::mat Pf = ksm_inp.Pf.slice(t);
    arma::mat Pp = ksm_inp.Pp.slice(t+1);
    Ji = Pf * Phi_tr * inv_sympd(Pp);

    arma::mat Jim_tr = Ji.t();

    xs_vals.col(t) = ksm_inp.xf.col(t) + Ji * (xs_vals.col(t+1) - ksm_inp.xp.col(t+1));
    Ps.slice(t) = Pf + Ji * (Ps.slice(t+1) - Pp) * Jim_tr;

    // smoothed Cov(x_t, x_t-1 | y_{1:T}): Needed for EM
    if (t > 0) {
      Jim_tr = ksm_inp.Pf.slice(t-1) * Phi_tr * inv_sympd(ksm_inp.Pp.slice(t));
      Plos.slice(t) = ksm_inp.Pf.slice(t) * Jim_tr +
        Ji * (Plos.slice(t+1) - ksm_inp.Phi * ksm_inp.Pf.slice(t)) * Jim_tr;
    }
  }

  // Smoothing t = 0
  Pp = ksm_inp.Pp.slice(0);
  Jim_tr = ksm_inp.P_0 * ksm_inp.Phi * inv_sympd(Pp);
  Plos.slice(0) = ksm_inp.Pf.slice(0) * Jim_tr.t() +
    Ji * (Plos.slice(1) - ksm_inp.Phi * ksm_inp.Pf.slice(0)) * Jim_tr.t();

  // Initial smoothed values
  arma::colvec x_0s = ksm_inp.x_0 + Jim_tr.t() * (xs_vals.col(0) - ksm_inp.xp.col(0));
  arma::mat P_0s = ksm_inp.P_0 + Jim_tr.t() * (Ps.slice(0) - Pp) * Jim_tr;


  return KalmanSmootherResult{
  .x_smoothed = xs_vals,
  .P_smoothed = Ps,
  .Lag_one_cov_smoothed = Plos,
  .x0_smoothed = x_0s,
  .P0_smoothed = P_0s
  };
}

KalmanSmootherResultMat FIS_cpp_mat(const KalmanSmootherInputMat& ksm_inp) {
  int p = ksm_inp.Phi.n_rows;
  int T = ksm_inp.xf.n_cols;
  int sym_len = p * (p + 1) / 2;

  arma::mat Ps(sym_len, T, arma::fill::zeros);
  arma::mat Plos(sym_len, T, arma::fill::zeros);

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


