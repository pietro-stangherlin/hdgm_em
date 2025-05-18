#include <armadillo>
#include <optional>
#include"Kalman_internal.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


// Change from github code: swap data matrix columns and rows definition
// so each observation is read by column (much more efficient in Armadillo)
// instead of by row

// Adding c++ classes to store c++ reading ready output
// for each cpp version define also the corresponding Rcpp version
// which ports the result to a Rcpp::list


// Implementation of Kalman filter
// X Data matrix (n x T)
// A Transition matrix (rp x rp)
// C Observation matrix (n x rp)
// Q State covariance (rp x rp)
// R Observation covariance (n x n)
// F_0 Initial state vector (rp x 1)
// P_0 Initial state covariance (rp x rp)
// retLL Return log-likelihood.
KalmanFilterResult SKF_cpp(arma::mat X,
                           arma::mat A,
                           arma::mat C,
                           arma::mat Q,
                           arma::mat R,
                           arma::colvec F_0,
                           arma::mat P_0,
                           bool retLL) {

  const int n = X.n_rows;
  const int T = X.n_cols;
  const int rp = A.n_rows;
  int n_c;


  // In internal code factors are Z (instead of F) and factor covariance V (instead of P),
  // to avoid confusion between the matrices and their predicted (p) and filtered (f) states.
  // Additionally the results matrices for all time periods have a T in the name.

  double loglik = retLL ? 0 : NA_REAL, dn = 0, detS;
  if(retLL) dn = n * log(2.0 * arma::datum::pi);
  arma::colvec Zp, Zf = F_0, et, xt;
  arma::mat K, Vp, Vf = P_0, S, VCt;

  // Predicted state mean and covariance
  arma::mat ZTp(rp, T, arma::fill::zeros);
  arma::cube VTp(rp, rp, T, arma::fill::zeros);

  // Filtered state mean and covariance
  arma::mat ZTf(rp, T, arma::fill::zeros);
  arma::cube VTf(rp, rp, T, arma::fill::zeros);

  // Handling missing values in the filter
  arma::mat Ci, Ri;
  arma::uvec nmiss, arow = find_finite(A.row(0));
  if(arow.n_elem == 0) Rcpp::stop("Missing first row of transition matrix");


  for (int i = 0; i < T; ++i) {


    // Run a prediction
    Zp = A * Zf;
    Vp = A * Vf * A.t() + Q;
    Vp += Vp.t(); // Ensure symmetry
    Vp *= 0.5;

    // If missing observations are present at some timepoints, exclude the
    // appropriate matrix slices from the filtering procedure.
    xt = X.col(i);
    nmiss = find_finite(xt);
    n_c = nmiss.n_elem;
    if(n_c > 0) {
      if(n_c == n) {
        Ci = C;
        Ri = R;
      } else {
        Ci = C.submat(nmiss, arow);
        Ri = R.submat(nmiss, nmiss);
        xt = xt.elem(nmiss);
      }


      // Intermediate results
      VCt = Vp * Ci.t();
      S = inv(Ci * VCt + Ri); // .i();


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
      if(retLL) {
        detS = det(S);
        if(detS > 0) loglik += log(detS) - arma::conv_to<double>::from(et.t() * S * et) - dn;
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


  if(retLL) loglik *= 0.5;

  return KalmanFilterResult{
    .F = ZTf, // filtered states
    .P = VTf, // filtered states Variances
    .F_pred = ZTp, // predicted states
    .P_pred = VTp, // predicted states variances
    .K_last = K, // Kalman gain last observation
    .C_last = Ci, // Observation matrix submatrix for non-missing (used in the smoother first step)
    .loglik = loglik
  };
}


// Runs a Kalman smoother
// A Transition matrix (rp x rp)
// ZTf State estimates
// ZTp State predicted estimates
// VTf_v Variance estimates
// VTp_v Predicted variance estimates
// K Kalman gain last observation
// Observation matrix submatrix for non-missing (used in the smoother first step)
// F_0 Initial state vector (rp x 1)
// P_0 Initial state covariance (rp x rp)
KalmanSmootherResult FIS_cpp(const arma::mat& A,
                             const arma::mat& ZTf,
                             const arma::mat& ZTp,
                             const arma::cube& VTf,
                             const arma::cube& VTp,
                             const arma::mat& K_last,
                             const arma::mat& C_last,
                             const arma::colvec& F_0,
                             const arma::mat& P_0) {
  const int T = ZTf.n_cols;
  const int rp = A.n_rows;

  arma::mat Vf = VTf.slice(T-1);
  arma::mat Vp = VTp.slice(T-1);
  arma::mat K; // Kalman gain last observation

  // Kalman smoothing
  arma::mat ZsT(rp, T, arma::fill::zeros);
  arma::cube VsT(rp, rp, T, arma::fill::zeros);
  arma::cube VVsT(rp, rp, T, arma::fill::zeros);

  ZsT.col(T-1) = ZTf.col(T-1);
  VsT.slice(T-1) = VTf.slice(T-1);

  arma::mat Ji, Jimt;
  arma::mat At = A.t();

  // Smoothed state variable and covariance
  for (int t = T - 1; t >= 0; --t) {
    arma::mat Vf = VTf.slice(t);
    arma::mat Vp = VTp.slice(t+1);
    Ji = Vf * At * inv_sympd(Vp);

    arma::mat Jimt = Ji.t();

    ZsT.col(t) = ZTf.col(t) + Ji * (ZsT.col(t+1) - ZTp.col(t+1));
    VsT.slice(t) = Vf + Ji * (VsT.slice(t+1) - Vp) * Jimt;

    // Cov(Z_t, Z_t-1): Needed for EM
    if (t > 0) {
      Jimt = VTf.slice(t-1) * At * inv_sympd(VTp.slice(t));
      VVsT.slice(t) = VTf.slice(t) * Jimt +
        Ji * (VVsT.slice(t+1) - A * VTf.slice(t)) * Jimt;
    }
  }

  // Smoothing t = 0
  Vp = VTp.slice(0);
  Jimt = P_0 * At * inv_sympd(Vp);
  VVsT.slice(0) = VTf.slice(0) * Jimt.t() +
    Ji * (VVsT.slice(1) - A * VTf.slice(0)) * Jimt.t();

  // Initial smoothed values
  arma::colvec F_smooth_0 = F_0 + Jimt * (ZsT.col(0) - ZTp.col(0));
  arma::mat P_smooth_0 = P_0 + Jimt * (VsT.slice(0) - Vp) * Jimt.t();


  return KalmanSmootherResult{
  .F_smooth = ZsT,
  .P_smooth = VsT,
  .Lag_one_cov_smooth = VVsT,
  .F_smooth_0 = F_smooth_0,
  .P_smooth_0 = P_smooth_0
  };
}

// Kalman Filter and Smoother
// X Data matrix (n x T)
// A Transition matrix (rp x rp)
// C Observation matrix (n x rp)
// Q State covariance (rp x rp)
// R Observation covariance (n x n)
// F_0 Initial state vector (rp x 1)
// P_0 Initial state covariance (rp x rp)
// retLL 0-no likelihood, 1-standard Kalman Filter, 2-BM14
KalmanFilterSmootherResult SKFS_cpp(const arma::mat& X,
                                    const arma::mat& A,
                                    const arma::mat& C,
                                    const arma::mat& Q,
                                    const arma::mat& R,
                                    const arma::colvec& F_0,
                                    const arma::mat& P_0,
                                    bool retLL) {

  KalmanFilterResult kf = SKF_cpp(X,A,C,Q,R,F_0,P_0,retLL);

  KalmanSmootherResult ks = FIS_cpp(A,kf.F,kf.F_pred,kf.P,kf.P_pred,
                                    kf.K_last,kf.C_last,
                                    F_0,P_0);



  KalmanFilterSmootherResult result;
  result.F = kf.F;
  result.P = kf.P;

  result.F_pred = kf.F_pred;
  result.P_pred = kf.P_pred;

  result.F_smooth = ks.F_smooth;
  result.P_smooth = ks.P_smooth;

  result.F_smooth_0 = ks.F_smooth_0;
  result.P_smooth_0 = ks.P_smooth_0;

  result.Lag_one_cov_smooth = ks.Lag_one_cov_smooth;
  if (retLL)
    result.loglik = kf.loglik;

  return result;
}

