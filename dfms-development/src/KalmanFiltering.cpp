#include <RcppArmadillo.h>
#include "helper.h"

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// Change from github code: swap data matrix columns and rows definition
// so each observation is read by column (much more efficient in Armadillo)
// instead of by row


// Implementation of a Kalman filter
// X Data matrix (n x T)
// A Transition matrix (rp x rp)
// C Observation matrix (n x rp)
// Q State covariance (rp x rp)
// R Observation covariance (n x n)
// F_0 Initial state vector (rp x 1)
// P_0 Initial state covariance (rp x rp)
// retLL Return log-likelihood.
// [[Rcpp::export]]
Rcpp::List SKF(arma::mat X, arma::mat A, arma::mat C, arma::mat Q,
               arma::mat R, arma::colvec F_0, arma::mat P_0, bool retLL = false) {

  const int n = X.n_rows;
  const int T = X.n_cols;
  const int rp = A.n_rows;
  int n_c;


  // In internal code factors are Z (instead of F) and factor covariance V (instead of P),
  // to avoid confusion between the matrices and their predicted (p) and filtered (f) states.
  // Additionally the results matrices for all time periods have a T in the name.

  double loglik = retLL ? 0 : NA_REAL, dn = 0, detS;
  if(retLL) dn = n * log(2.0 * datum::pi);
  colvec Zp, Zf = F_0, et, xt;
  mat K, Vp, Vf = P_0, S, VCt;

  // Predicted state mean and covariance
  mat ZTp(rp, T, fill::zeros);
  cube VTp(rp, rp, T, fill::zeros);

  // Filtered state mean and covariance
  mat ZTf(rp, T, fill::zeros);
  cube VTf(rp, rp, T, fill::zeros);

  // Handling missing values in the filter
  mat Ci, Ri;
  uvec nmiss, arow = find_finite(A.row(0));
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
        if(detS > 0) loglik += log(detS) - conv_to<double>::from(et.t() * S * et) - dn;
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

  return Rcpp::List::create(Rcpp::Named("F") = ZTf,
                            Rcpp::Named("P") = VTf,
                            Rcpp::Named("F_pred") = ZTp,
                            Rcpp::Named("P_pred") = VTp,
                            // Rcpp::Named("F_0") = F_0,
                            // Rcpp::Named("P_0") = P_0,
                            Rcpp::Named("loglik") = loglik);
}

// Runs a Kalman smoother
// A Transition matrix (rp x rp)
// ZTf State estimates
// ZTp State predicted estimates
// VTf_v Variance estimates
// VTp_v Predicted variance estimates
// F_0 Initial state vector (rp x 1)
// P_0 Initial state covariance (rp x rp)
// [[Rcpp::export]]
Rcpp::List FIS(arma::mat A,
               arma::mat ZTf, arma::mat ZTp,
               Rcpp::NumericVector VTf_v,
               Rcpp::NumericVector VTp_v,
               SEXP F_0SEXP, SEXP P_0SEXP) {

  const int T = ZTf.n_cols;
  const int rp = A.n_rows;

  arma::cube VTf = array2cube(VTf_v);
  arma::cube VTp = array2cube(VTp_v);

  // Smoothed state mean and covariance
  arma::mat ZsT(rp, T, fill::zeros);
  arma::cube VsT(rp, rp, T, fill::zeros);

  // Initialize smoothed values at T-1 with filtered estimates
  ZsT.col(T-1) = ZTf.col(T-1);
  VsT.slice(T-1) = VTf.slice(T-1);

  arma::mat At = A.t(), Ji, Vfi, Vpi;

  // Backward smoothing loop
  // See e.g. Shumway and Stoffer (2002) p297, or astsa::Ksmooth0
  for (int i = T-2; i >= 0; --i) {
    Vfi = VTf.slice(i);
    Vpi = VTp.slice(i+1);
    Ji = Vfi * At * inv_sympd(Vpi);
    ZsT.col(i) = ZTf.col(i) + Ji * (ZsT.col(i+1) - ZTp.col(i+1));
    VsT.slice(i) = Vfi + Ji * (VsT.slice(i+1) - Vpi) * Ji.t();
  }

  if (Rf_isNull(F_0SEXP) || Rf_isNull(P_0SEXP)) {
    return Rcpp::List::create(
      Rcpp::Named("F_smooth") = ZsT,
      Rcpp::Named("P_smooth") = VsT
    );
  }

  // Optional: smooth initial state
  Rcpp::NumericVector F_0_v(F_0SEXP);
  Rcpp::NumericMatrix P_0_v(P_0SEXP);
  int n = P_0_v.nrow(), k = P_0_v.ncol();
  arma::mat P_0(P_0_v.begin(), n, k, false);
  arma::colvec F_0(F_0_v.begin(), n, false);

  Vpi = VTp.slice(0);
  Ji = P_0 * At * inv_sympd(Vpi);

  return Rcpp::List::create(
    Rcpp::Named("F_smooth") = ZsT,
    Rcpp::Named("P_smooth") = VsT,
    Rcpp::Named("F_smooth_0") = F_0 + Ji * (ZsT.col(0) - ZTp.col(0)),
    Rcpp::Named("P_smooth_0") = P_0 + Ji * (VsT.slice(0) - Vpi) * Ji.t()
  );
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
// [[Rcpp::export]]
Rcpp::List SKFS(arma::mat &X, arma::mat A, arma::mat C, arma::mat Q,
                arma::mat R, arma::colvec F_0, arma::mat P_0, bool retLL = false) {

  const int n = X.n_rows;      // variables
  const int T = X.n_cols;      // time steps
  const int rp = A.n_rows;     // state dimension

  double loglik = retLL ? 0.0 : NA_REAL;
  const double dn = retLL ? n * std::log(2.0 * datum::pi) : 0.0;

  arma::colvec Zf = F_0, Zp, xt, et;
  arma::mat Vf = P_0, Vp, S, VCt, K;

  // Predicted and filtered states (rp × T)
  arma::mat ZTp(rp, T, fill::zeros);
  arma::cube VTp(rp, rp, T, fill::zeros);

  arma::mat ZTf(rp, T, fill::zeros);
  arma::cube VTf(rp, rp, T, fill::zeros);

  // Kalman filtering loop
  for (int t = 0; t < T; ++t) {

    Zp = A * Zf;
    Vp = A * Vf * A.t() + Q;
    Vp = 0.5 * (Vp + Vp.t());  // ensure symmetry

    xt = X.col(t);  // n x 1
    arma::uvec obs_idx = arma::find_finite(xt);
    int n_obs = obs_idx.n_elem;

    if (n_obs > 0) {
      arma::mat Ci;
      arma::mat Ri;
      arma::colvec xt_obs;

      if (n_obs == n) {
        Ci = C;
        Ri = R;
        xt_obs = xt;
      } else {
        Ci = C.rows(obs_idx);
        Ri = R.submat(obs_idx, obs_idx);
        xt_obs = xt.elem(obs_idx);
      }

      VCt = Vp * Ci.t();             // rp × obs
      S = Ci * VCt + Ri;             // obs × obs
      arma::mat Sinv = inv_sympd(S);
      et = xt_obs - Ci * Zp;
      K = VCt * Sinv;                // rp × obs

      Zf = Zp + K * et;              // rp × 1
      Vf = Vp - K * Ci * Vp;
      Vf = 0.5 * (Vf + Vf.t());
      if (retLL) {
        double detS = arma::det(S);
        if (detS > 0.0)
          loglik += std::log(detS) + arma::as_scalar(et.t() * Sinv * et);
      }
    } else {
      Zf = Zp;
      Vf = Vp;
    }

    ZTp.col(t) = Zp;
    VTp.slice(t) = Vp;
    ZTf.col(t) = Zf;
    VTf.slice(t) = Vf;
  }

  if (retLL) loglik = -0.5 * loglik;

  // Kalman smoothing
  arma::mat ZsT(rp, T, fill::zeros);
  arma::cube VsT(rp, rp, T, fill::zeros);
  arma::cube VVsT(rp, rp, T, fill::zeros);

  ZsT.col(T-1) = ZTf.col(T-1);
  VsT.slice(T-1) = VTf.slice(T-1);

  arma::mat Ji, Jimt;

  for (int t = T - 2; t >= 0; --t) {
    arma::mat Vf = VTf.slice(t);
    arma::mat Vp = VTp.slice(t+1);
    Ji = Vf * A.t() * inv_sympd(Vp);

    ZsT.col(t) = ZTf.col(t) + Ji * (ZsT.col(t+1) - ZTp.col(t+1));
    VsT.slice(t) = Vf + Ji * (VsT.slice(t+1) - Vp) * Ji.t();

    if (t > 0) {
      Jimt = VTf.slice(t-1) * A.t() * inv_sympd(VTp.slice(t));
      VVsT.slice(t) = VTf.slice(t) * Jimt.t() +
        Ji * (VVsT.slice(t+1) - A * VTf.slice(t)) * Jimt.t();
    }
  }

  // Smoothing t = 0
  Vp = VTp.slice(0);
  Jimt = P_0 * A.t() * inv_sympd(Vp);
  VVsT.slice(0) = VTf.slice(0) * Jimt.t() +
    Ji * (VVsT.slice(1) - A * VTf.slice(0)) * Jimt.t();

  // Initial smoothed values
  arma::colvec F_smooth_0 = F_0 + Jimt * (ZsT.col(0) - ZTp.col(0));
  arma::mat P_smooth_0 = P_0 + Jimt * (VsT.slice(0) - Vp) * Jimt.t();

  return Rcpp::List::create(
    Rcpp::Named("F") = ZTf,
    Rcpp::Named("P") = VTf,
    Rcpp::Named("F_pred") = ZTp,
    Rcpp::Named("P_pred") = VTp,
    Rcpp::Named("F_smooth") = ZsT,
    Rcpp::Named("P_smooth") = VsT,
    Rcpp::Named("PPm_smooth") = VVsT,
    Rcpp::Named("F_smooth_0") = F_smooth_0,
    Rcpp::Named("P_smooth_0") = P_smooth_0,
    Rcpp::Named("loglik") = loglik
  );
}






