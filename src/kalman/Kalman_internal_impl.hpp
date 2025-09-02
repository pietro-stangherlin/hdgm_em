#pragma once

#include <RcppArmadillo.h>
#include <type_traits>
#include <iostream>
#include <cstdlib> // For atoi

#include "Kalman_internal.hpp"
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
  arma::mat I(p, p, arma::fill::eye);
  arma::mat temp_matr, temp_chol, S_chol;
  arma::vec temp_diag;
  double log_sum_diag = 0.0;

  xft = kf_inp.x_0;
  Pft = kf_inp.P_0;

  // define once
  arma::mat Phi_tr = kf_inp.Phi.t();

  arma::uvec nmiss, arow = arma::find_finite(kf_inp.Phi.row(0));
  int n_c = 0; // used to count number of missing observations at each time
  double loglik = kf_inp.retLL ? -q * std::log(2 * 3.14159265359) : std::numeric_limits<double>::quiet_NaN();

  for (int t = 0; t < T; ++t) {
    xpt = kf_inp.Phi * xft;
    Ppt = kf_inp.Phi * Pft * Phi_tr + kf_inp.Q;
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
      temp_matr = At * VCt + Rt;
      // to check code below
      // temp_matr = 0.5 * (temp_matr + temp_matr.t());
      // temp_chol = arma::chol(temp_matr, "lower");
      // S_chol = arma::inv(arma::trimatl(temp_chol));
      // S = S_chol * S_chol.t();
      S = arma::inv(At * VCt + Rt);

      // Prediction error
      et = yt - At * xpt;


      // Kalman gain
      K = VCt * S;
      // Updated state estimate
      xft = xpt + K * et;
      // Updated state covariance estimate
      Pft = (I - K * At) * Ppt;
      Pft = (Pft + Pft.t()) * 0.5; // Ensure symmetry

      if (kf_inp.retLL) {
        // since S = S_chol * S_chol.t()
        // det(S) = det(S_chol)det(S_chol.t()) = det(S_chol)^2
        // and log(det(S_chol)^2) = 2 * log(det(S_chol))
        // since S_chol is lower triangular:
        // 2 * log(det(S_chol)) = 2 * log(prod(diag(S_chol))) = 2 * sum(log(diag(S_chol)))
        // temp_diag = S_chol.diag();
        // for(int j = 0; j < temp_diag.n_elem; j++){
        //   log_sum_diag += std::log(temp_diag[j]);
        // }

        double log_det_val, det_sign;
        //log_det_val = 2 * log_sum_diag;
        arma::log_det(log_det_val, det_sign, S);

        if (det_sign > 0) loglik += log_det_val - arma::as_scalar(et.t() * S * et);

        // reset
        log_sum_diag = 0.0;
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

// ---------------------  Kalman Filter ---------------------- //

// In future this has to be included in SKF_core as a template
// in order to have just one template
template <typename CovStore>
KalmanFilterResultT<CovStore> SKF_core_TimeVaryingObsMatr(const KalmanFilterInputTimeVaryingObsMatr& kf_inp,
                                                          CovStore& Pp_store, CovStore& Pf_store) {

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
  arma::mat I(p, p, arma::fill::eye);
  arma::mat temp_matr, temp_chol, S_chol;
  arma::vec temp_diag;
  double log_sum_diag = 0.0;

  xft = kf_inp.x_0;
  Pft = kf_inp.P_0;

  // define once
  arma::mat Phi_tr = kf_inp.Phi.t();

  arma::uvec nmiss, arow = arma::find_finite(kf_inp.Phi.row(0));
  int n_c = 0; // used to count number of missing observations at each time
  double loglik = kf_inp.retLL ? -q * std::log(2 * 3.14159265359) : std::numeric_limits<double>::quiet_NaN();

  for (int t = 0; t < T; ++t) {
    xpt = kf_inp.Phi * xft;
    Ppt = kf_inp.Phi * Pft * Phi_tr + kf_inp.Q;
    Ppt = 0.5 * (Ppt + Ppt.t()); // force symmetry

    // DEBUG
    if(Ppt.is_sympd() != true){
      std::cout << "Warning: Ppt is NOT positive definite; " << "time: " << t << std::endl;
    }

    At = kf_inp.A_cube.slice(t);

    yt = kf_inp.Y.col(t);
    nmiss = arma::find_finite(yt);

    n_c = nmiss.n_elem;

    if (n_c > 0) {
      if (n_c == q) {
        At = At;
        Rt = kf_inp.R;
      } else {
        At = At.submat(nmiss, arow);
        Rt = kf_inp.R.submat(nmiss, nmiss);
        yt = yt.elem(nmiss);
      }

      // Intermediate results
      VCt = Ppt * At.t();
      temp_matr = At * VCt + Rt;
      // to check code below
      // temp_matr = 0.5 * (temp_matr + temp_matr.t());
      // temp_chol = arma::chol(temp_matr, "lower");
      // S_chol = arma::inv(arma::trimatl(temp_chol));
      // S = S_chol * S_chol.t();
      S = arma::inv(At * VCt + Rt);

      // Prediction error
      et = yt - At * xpt;


      // Kalman gain
      K = VCt * S;
      // Updated state estimate
      xft = xpt + K * et;
      // Updated state covariance estimate
      Pft = (I - K * At) * Ppt;
      Pft = (Pft + Pft.t()) * 0.5; // Ensure symmetry

      if (kf_inp.retLL) {
        // since S = S_chol * S_chol.t()
        // det(S) = det(S_chol)det(S_chol.t()) = det(S_chol)^2
        // and log(det(S_chol)^2) = 2 * log(det(S_chol))
        // since S_chol is lower triangular:
        // 2 * log(det(S_chol)) = 2 * log(prod(diag(S_chol))) = 2 * sum(log(diag(S_chol)))
        // temp_diag = S_chol.diag();
        // for(int j = 0; j < temp_diag.n_elem; j++){
        //   log_sum_diag += std::log(temp_diag[j]);
        // }

        double log_det_val, det_sign;
        //log_det_val = 2 * log_sum_diag;
        arma::log_det(log_det_val, det_sign, S);

        if (det_sign > 0) loglik += log_det_val - arma::as_scalar(et.t() * S * et);

        // reset
        log_sum_diag = 0.0;
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

  arma::mat J_t, J_t_T;
  arma::mat Phi_tr = ksm_inp.Phi.t();

  // populate last smoothed values
  xs_vals.col(T-1) = ksm_inp.xf.col(T-1); // last smoothed state = filtered state
  K_times_A = (ksm_inp.nc_last == 0) ? arma::mat(p, p, arma::fill::zeros) : ksm_inp.K_last * ksm_inp.A_last;

  Ps_t = GetCov(ksm_inp.Pf, T-1, p); // last smoothed state cov = filtered state cov


  Plos_t = (arma::eye(p,p) - K_times_A) * ksm_inp.Phi * GetCov(ksm_inp.Pf, T-2, p);

  if constexpr (std::is_same_v<CovStore, arma::cube>) {
    Ps.slice(T-1) = Ps_t;
    Plos.slice(T-2) = Plos_t;
  } else {
    Ps.col(T-1) = FromSymMatrixToVector(Ps_t);
    Plos.col(T-2) = FromSymMatrixToVector(Plos_t);
  };

  // Smoothed state variable and covariance
  for (int t = T - 2; t >= 0; --t) {
    arma::mat Pf = GetCov(ksm_inp.Pf, t, p);
    arma::mat Pp = GetCov(ksm_inp.Pp, t+1, p);


    // DEBUG
    // std::cout << "Pp" <<  std::endl;
    // std::cout << Pp << std::endl;
    J_t = Pf * Phi_tr * arma::inv_sympd(Pp);

    arma::mat J_t_T = J_t.t();

    xs_vals.col(t) = ksm_inp.xf.col(t) + J_t * (xs_vals.col(t+1) - ksm_inp.xp.col(t+1));
    Ps_t = Pf + J_t * (GetCov(Ps, t+1, p) - Pp) * J_t_T;

    if constexpr (std::is_same_v<CovStore, arma::cube>) {
      Ps.slice(t) = Ps_t;
    } else {
      Ps.col(t) = FromSymMatrixToVector(Ps_t);
    };

    // smoothed Cov(x_t, x_t-1 | y_{1:T}): Needed for EM
    // NOTE: watch out about the different indexes (excluding zero states)
    // between Filtered, Predicted and Smoothed Covariances: times = 1,..,T (array indexes from 0 to T-1)
    // and Lag One Smoothed Covariances: times = 2,...,T (array indexes from 0 to T-2)
    // since in C++ indexes starts from 0:
    // T - 1,   T - 2,  T - 3,...,(array indexes other)
    // T - 2,   T - 3,  T - 4,..., (array indexes Lag-one)
    // (T, T-1), (T-1, T-2), () (actual times indexes Lag-one)


    // so t = 1 array index here is used to store actual times (2,1) lag one smoother
    // while t = 0 for (1, 0) (defined out of the loop)
    if (t > 1) {

      // here index t is referred to NOT lag one smoothed covariances

      J_t_T = arma::inv_sympd(GetCov(ksm_inp.Pp, t-1, p)) * ksm_inp.Phi * GetCov(ksm_inp.Pf, t-2, p);
      Plos_t = GetCov(ksm_inp.Pf,t, p) * J_t_T +
        J_t * (GetCov(Plos, t, p) - ksm_inp.Phi * GetCov(ksm_inp.Pf, t, p)) * J_t_T;

      if constexpr (std::is_same_v<CovStore, arma::cube>) {
        Plos.slice(t-1) = Plos_t;
      } else {
        Plos.col(t-1) = FromSymMatrixToVector(Plos_t);
      };

    }
  }

  // Smoothing t = 0 (array index) (actual time = 1)

  Pp = GetCov(ksm_inp.Pp, 0, p);

  J_t_T = arma::inv_sympd(Pp) * ksm_inp.Phi *  ksm_inp.P_0;
  Plos_t = GetCov(ksm_inp.Pf, 0, p) * J_t_T +
    J_t * (GetCov(Plos, 1, p) - ksm_inp.Phi * GetCov(ksm_inp.Pf, 0, p)) * J_t_T;

  if constexpr (std::is_same_v<CovStore, arma::cube>) {
    Plos.slice(0) = Plos_t;
  } else {
    Plos.col(0) = FromSymMatrixToVector(Plos_t);
  };

  // Initial smoothed values
  arma::colvec x_0s = ksm_inp.x_0 + J_t_T.t() * (xs_vals.col(0) - ksm_inp.xp.col(0));
  arma::mat P_0s = ksm_inp.P_0 + J_t_T.t() * (GetCov(Ps, 0, p) - Pp) * J_t_T;


  return KalmanSmootherResultT<CovStore>{
    .x_smoothed = xs_vals,
    .P_smoothed = Ps,
    .Lag_one_cov_smoothed = Plos,
    .x0_smoothed = x_0s,
    .P0_smoothed = P_0s
  };
}

