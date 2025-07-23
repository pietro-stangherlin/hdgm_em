#pragma once

#include <optional>
#include <RcppArmadillo.h>
#include <iostream>
#include <cstdlib> // For atoi

#include "Kalman_internal.h"
#include "../utils/symmetric_matr_vec.h"

// template with options to use arma::cube or arma::mat (vectorized) (symmetric: half space)
// to store the filtered and predicted state covariances
// Pp_store: predicted state covariance matrices
// Pf_store: filtered state covariance matrices
template <typename CovStore>
KalmanFilterResultT<CovStore> SKF_core(const KalmanFilterInput& kf_inp, CovStore& Pp_store, CovStore& Pf_store) {


  const int q = kf_inp.Y.n_rows; // observation vector dimensions
  const int p = kf_inp.Phi.n_rows; // state vector dimensions
  const int T = kf_inp.Y.n_cols; // number of observations

  // predicted and filtered state (mean)
  arma::mat xp_vals(p, T, arma::fill::zeros);
  arma::mat xf_vals(p, T, arma::fill::zeros);

  // current iteration tempory values
  arma::vec xpt, xft, yt, et;
  arma::mat K, Ppt, Pft, S, VCt;
  arma::mat At, Rt;


  xft = kf_inp.x_0;
  Pft = kf_inp.P_0;

  arma::uvec nmiss, arow = arma::find_finite(kf_inp.Phi.row(0));
  int n_c = 0; // used to count number of missing observations at each time
  double loglik = kf_inp.retLL ? 0.0 : std::numeric_limits<double>::quiet_NaN();
  double dn = 0.0;

  for (int t = 0; t < T; ++t) {
    xpt = kf_inp.Phi * xft;
    Ppt = kf_inp.Phi * Pft * kf_inp.Phi.t() + kf_inp.Q;
    Ppt = 0.5 * (Ppt + Ppt.t()); // force symmetry

    yt = kf_inp.Y.col(t);
    nmiss = arma::find_finite(yt);
    n_c = nmiss.n_elem;

    if (n_c > 0) {
      if (n_c == q) {
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

      if (kf_inp.retLL) {
        //TO DO: Maybe convenient to compute one S decomposition
        // once and then do inverse and det
        double log_det_val, det_sign;
        arma::log_det(log_det_val, det_sign, S);
        if (det_sign > 0) loglik += log_det_val - arma::as_scalar(et.t() * S * et) - dn;
      }

      // If all missing: just prediction.
    } else {
      xft = xpt;
      Pft = Ppt;
    }

    // Store predicted and filtered data needed for smoothing
    xf_vals.col(t) = xft;
    xp_vals.col(t) = xpt;

    // Store covariance
    if constexpr (std::is_same_v<CovStore, arma::cube>) {
      Pp_store.slice(t) = Ppt;
      Pf_store.slice(t) = Pft;
    } else {
      Pp_store.col(t) = FromSymMatrixToVector(Ppt);
      Pf_store.col(t) = FromSymMatrixToVector(Pft);
    }
  }

  if (kf_inp.retLL) loglik *= 0.5;

  return KalmanFilterResultT<CovStore>{
    .xf = xf_vals,
    .Pf = Pf_store,
    .xp = xp_vals,
    .Pp = Pp_store,
    .K_last = K,
    .A_last = At,
    .nc_last = n_c,
    .loglik = loglik
  };
}
