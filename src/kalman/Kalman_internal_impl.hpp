#pragma once

#include <RcppArmadillo.h>
#include <optional>
#include <type_traits>
#include <iostream>
#include <cstdlib> // For atoi

#include "Kalman_internal.h"
#include "../utils/symmetric_matr_vec.h"


// ---------------------  Kalman Filter ---------------------- //

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



// ---------------------  Kalman Smoother ---------------------- //
// NOTE: here is assumed that:
// if KalmanFilter Input is stored as matrices -> KalmanSmoother Input and Output are stored as matrices
// if KalmanFilter Input is stored as cubes -> KalmanSmoother Input and Output are stored as cubes
// NO "cross" matrix-cube compatibility

// for parameters description see Kalman_types.h
template <typename CovStore>
KalmanSmootherResultT<CovStore> FIS_core(const KalmanSmootherInputT<CovStore>& ksm_inp, CovStore& Ps, CovStore& Plos) {

  const int T = ksm_inp.xf.n_cols;
  const int p = ksm_inp.Phi.n_rows;

  arma::mat Pf = GetCov(ksm_inp.Pf, T-1, p);
  arma::mat Pp = GetCov(ksm_inp.Pp, T-1, p);
  arma::mat K_times_A; // Kalman gain last observation times observation matrix

  // allocate smoothed quantities
  arma::mat xs_vals(p, T, arma::fill::zeros);

  arma::mat Ps_t;
  arma::mat Plos_t;

  arma::mat Ji, Jim_tr;
  arma::mat Phi_tr = ksm_inp.Phi.t();

  // populate last smoothed values
  xs_vals.col(T-1) = ksm_inp.xf.col(T-1); // last smoothed state = filtered state
  K_times_A = (ksm_inp.nc_last == 0) ? arma::mat(p, p, arma::fill::zeros) : ksm_inp.K_last * ksm_inp.A_last;


  Ps_t = GetCov(ksm_inp.Pf, T-1, p); // last smoothed state cov = filtered state cov

  Plos_t = (arma::eye(p,p) - K_times_A) * ksm_inp.Phi * GetCov(ksm_inp.Pf, T-2, p);

  if constexpr (std::is_same_v<CovStore, arma::cube>) {
    Ps.slice(T-1) = Ps_t;
    Plos.slice(T-1) = Plos_t;
  } else {
    Ps.col(T-1) = FromSymMatrixToVector(Ps_t);
    Plos.col(T-1) = FromSymMatrixToVector(Plos_t);
  };

  // Smoothed state variable and covariance
  for (int t = T - 2; t >= 0; --t) {
    arma::mat Pf = GetCov(ksm_inp.Pf, t, p);
    arma::mat Pp = GetCov(ksm_inp.Pp, t+1, p);

    Ji = Pf * Phi_tr * inv_sympd(Pp);

    arma::mat Jim_tr = Ji.t();

    xs_vals.col(t) = ksm_inp.xf.col(t) + Ji * (xs_vals.col(t+1) - ksm_inp.xp.col(t+1));
    Ps_t = Pf + Ji * (GetCov(Ps, t+1, p) - Pp) * Jim_tr;

    if constexpr (std::is_same_v<CovStore, arma::cube>) {
      Ps.slice(t) = Ps_t;
    } else {
      Ps.col(t) = FromSymMatrixToVector(Ps_t);
    };

    // smoothed Cov(x_t, x_t-1 | y_{1:T}): Needed for EM
    if (t > 0) {

      Jim_tr = GetCov(ksm_inp.Pf, t-1, p) * Phi_tr * arma::inv_sympd(GetCov(ksm_inp.Pp, t, p));
      Plos_t = GetCov(ksm_inp.Pf,t, p) * Jim_tr +
        Ji * (GetCov(Plos, t+1, p) - ksm_inp.Phi * GetCov(ksm_inp.Pf, t, p)) * Jim_tr;

      if constexpr (std::is_same_v<CovStore, arma::cube>) {
        Plos.slice(t) = Plos_t;
      } else {
        Plos.col(t) = FromSymMatrixToVector(Plos_t);
      };

    }
  }

  // Smoothing t = 0
  Pp = GetCov(ksm_inp.Pp, 0, p);
  Jim_tr = ksm_inp.P_0 * Phi_tr * inv_sympd(Pp);
  Plos_t = GetCov(ksm_inp.Pf, 0, p) * Jim_tr.t() +
    Ji * (GetCov(Plos, 1, p) - ksm_inp.Phi * GetCov(ksm_inp.Pf, 0, p)) * Jim_tr.t();

  if constexpr (std::is_same_v<CovStore, arma::cube>) {
    Plos.slice(0) = Plos_t;
  } else {
    Plos.col(0) = FromSymMatrixToVector(Plos_t);
  };

  // Initial smoothed values
  arma::colvec x_0s = ksm_inp.x_0 + Jim_tr * (xs_vals.col(0) - ksm_inp.xp.col(0));
  arma::mat P_0s = ksm_inp.P_0 + Jim_tr * (GetCov(Ps, 0, p) - Pp) * Jim_tr.t();


  return KalmanSmootherResultT<CovStore>{
    .x_smoothed = xs_vals,
    .P_smoothed = Ps,
    .Lag_one_cov_smoothed = Plos,
    .x0_smoothed = x_0s,
    .P0_smoothed = P_0s
  };
}

