#include <optional>
#include <RcppArmadillo.h>
#include <iostream>
#include <cstdlib> // For atoi

#include"Kalman_internal.h"

// This code is an adaptaion from Sebastian Krantz DFMS package.
// https://cran.r-project.org/package=dfms

/* Linear Gaussian State-Space Kalman Filter and Smoother implementations.
 * With reference to the Shumway and Stoffer model here
 * - obsrvations and state covariance matrices are considered constant in time
 * - no exogenus variables are considered
 * (one can always consider the residuals as the new response)
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

  const int n = kf_inp.X.n_rows;
  const int T = kf_inp.X.n_cols;
  const int rp = kf_inp.A.n_rows;
  int n_c;

  //std::cout << "X dims: " << n << " x " << T << std::endl;

  // In internal code factors are Z (instead of F) and factor covariance V (instead of P),
  // to avoid confusion between the matrices and their predicted (p) and filtered (f) states.
  // Additionally the results matrices for all time periods have a T in the name.

  double loglik = kf_inp.retLL ? 0.0 : std::numeric_limits<double>::quiet_NaN();

  double dn = 0.0;
  double detS = 0.0;

  //std::cout << "AFTER llik;\n";

  arma::vec Zp, Zf, et, xt;
  Zf = kf_inp.F_0;


  arma::mat K, Vp, Vf, S, VCt;
  Vf = kf_inp.P_0;

  // Predicted state mean and covariance
  arma::mat ZTp(rp, T, arma::fill::zeros);
  arma::cube VTp(rp, rp, T, arma::fill::zeros);

  // Filtered state mean and covariance
  arma::mat ZTf(rp, T, arma::fill::zeros);
  arma::cube VTf(rp, rp, T, arma::fill::zeros);

  // Handling missing values in the filter
  arma::mat Ci, Ri;
  arma::uvec nmiss, arow = arma::find_finite(kf_inp.A.row(0));
  if (arow.n_elem == 0) {
    throw std::runtime_error("Missing first row of transition matrix\n");
  }


  //std::cout << "[DEBUG] Before for loop" << std::endl;
  for (int i = 0; i < T; ++i) {

    // Run a prediction

    Zp = kf_inp.A * Zf;
    Vp = kf_inp.A * Vf * kf_inp.A.t() + kf_inp.Q;
    Vp += Vp.t(); // Ensure symmetry
    Vp *= 0.5;

    // If missing observations are present at some timepoints, exclude the
    // appropriate matrix slices from the filtering procedure.
    xt = kf_inp.X.col(i);
    nmiss = arma::find_finite(xt);
    n_c = nmiss.n_elem;
    if(n_c > 0) {
      if(n_c == n) {
        Ci = kf_inp.C;
        Ri = kf_inp.R;
      } else {
        Ci = kf_inp.C.submat(nmiss, arow);
        Ri = kf_inp.R.submat(nmiss, nmiss);
        xt = xt.elem(nmiss);
      }

      // Intermediate results
      VCt = Vp * Ci.t();
      S = arma::inv_sympd(Ci * VCt + Ri);


      // Prediction error
      et = xt - Ci * Zp;
      // Kalman gain
      K = VCt * S;
      // Updated state estimate
      Zf = Zp + K * et;
      // Updated state covariance estimate
      Vf = Vp - K * Ci * Vp;
      Vf += Vf.t(); // Ensure symmetry
      Vf *= 0.5;

      // Compute likelihood. Skip this part if S is not positive definite.
      if(kf_inp.retLL) {
        //TO DO: Maybe convenient to compute one decomposition
        // once and then do inverse and det
        double log_det_val;
        double det_sign;
        arma::log_det(log_det_val, det_sign, S);
        if(det_sign > 0) loglik += log_det_val - arma::as_scalar(et.t() * S * et) - dn;

      }

    } else { // If all missing: just prediction.
      Zf = Zp;
      Vf = Vp;
    }


    // Store predicted and filtered data needed for smoothing
    ZTp.col(i) = Zp;
    VTp.slice(i) = Vp;
    ZTf.col(i) = Zf;
    VTf.slice(i) = Vf;

  }


  if(kf_inp.retLL) loglik *= 0.5;

  return KalmanFilterResult{
    .F = ZTf, // filtered states
    .P = VTf, // filtered states Variances
    .F_pred = ZTp, // predicted states
    .P_pred = VTp, // predicted states variances
    .K_last = K, // Kalman gain last observation
    .C_last = Ci, // Observation matrix submatrix for non-missing (used in the smoother first step)
    .nc_last = n_c,
    .loglik = loglik
  };
}


// Kalman Smoother
// for parameters description see Kalman_types.h
KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp) {

  // std::cout << "Inside FIS_cpp" << std::endl;

  const int T = ksm_inp.ZTf.n_cols;
  const int rp = ksm_inp.A.n_rows;

  arma::mat Vf = ksm_inp.VTf.slice(T-1);
  arma::mat Vp = ksm_inp.VTp.slice(T-1);
  arma::mat K; // Kalman gain last observation

  // Kalman smoothing
  arma::mat ZsT(rp, T, arma::fill::zeros);
  arma::cube VsT(rp, rp, T, arma::fill::zeros);
  arma::cube VVsT(rp, rp, T, arma::fill::zeros);


  // populate last smoothed values
  ZsT.col(T-1) = ksm_inp.ZTf.col(T-1); // last smoothed state = filtered state
  VsT.slice(T-1) = ksm_inp.VTf.slice(T-1); // last smoothed state cov = filtered state cov


  arma::mat Ji, Jimt;
  arma::mat At = ksm_inp.A.t();

  K = (ksm_inp.nc_last == 0) ? arma::mat(rp, rp, arma::fill::zeros) : ksm_inp.K_last * ksm_inp.C_last;

  VVsT.slice(T-1) = (arma::eye(rp,rp) - K) * ksm_inp.A * ksm_inp.VTf.slice(T-2);

  // Smoothed state variable and covariance
  for (int t = T - 2; t >= 0; --t) {
    arma::mat Vf = ksm_inp.VTf.slice(t);
    arma::mat Vp = ksm_inp.VTp.slice(t+1);
    Ji = Vf * At * inv_sympd(Vp);

    arma::mat Jimt = Ji.t();

    ZsT.col(t) = ksm_inp.ZTf.col(t) + Ji * (ZsT.col(t+1) - ksm_inp.ZTp.col(t+1));
    VsT.slice(t) = Vf + Ji * (VsT.slice(t+1) - Vp) * Jimt;

    // Cov(Z_t, Z_t-1): Needed for EM
    if (t > 0) {
      Jimt = ksm_inp.VTf.slice(t-1) * At * inv_sympd(ksm_inp.VTp.slice(t));
      VVsT.slice(t) = ksm_inp.VTf.slice(t) * Jimt +
        Ji * (VVsT.slice(t+1) - ksm_inp.A * ksm_inp.VTf.slice(t)) * Jimt;
    }
  }

  // Smoothing t = 0
  Vp = ksm_inp.VTp.slice(0);
  Jimt = ksm_inp.P_0 * At * inv_sympd(Vp);
  VVsT.slice(0) = ksm_inp.VTf.slice(0) * Jimt.t() +
    Ji * (VVsT.slice(1) - ksm_inp.A * ksm_inp.VTf.slice(0)) * Jimt.t();

  // Initial smoothed values
  arma::colvec F_smooth_0 = ksm_inp.F_0 + Jimt * (ZsT.col(0) - ksm_inp.ZTp.col(0));
  arma::mat P_smooth_0 = ksm_inp.P_0 + Jimt * (VsT.slice(0) - Vp) * Jimt.t();


  return KalmanSmootherResult{
  .F_smooth = ZsT,
  .P_smooth = VsT,
  .Lag_one_cov_smooth = VVsT,
  .F_smooth_0 = F_smooth_0,
  .P_smooth_0 = P_smooth_0
  };
}

// Kalman Filter and Smoother
// Only Kalman Smoother ouptput is returned
// for parameters description see Kalman_types.h
KalmanSmootherResult SKFS_cpp(const KalmanFilterInput& kfsm_inp) {

  KalmanFilterResult kf = SKF_cpp(kfsm_inp);

  KalmanSmootherInput ksmin = {
  .A = kfsm_inp.A,
  .ZTf = kf.F,
  .ZTp = kf.F_pred,
  .VTf = kf.P,
  .VTp = kf.P_pred,
  .K_last = kf.K_last,
  .C_last = kf.C_last,
  .F_0 = kfsm_inp.F_0,
  .P_0 = kfsm_inp.P_0,
  .nc_last = kf.nc_last,
  };

  //std::cout << "Kalman Filter finished\n";

  KalmanSmootherResult ksmout = FIS_cpp(ksmin);

  //std::cout << "Kalman Smoother finished\n";

  return ksmout;
}



