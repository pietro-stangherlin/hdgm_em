#include <RcppArmadillo.h>
#include "Kalman_internal.h"


// R version
// [[Rcpp::export]]
Rcpp::List SKF(const arma::mat& X,
               const arma::mat& A,
               const arma::mat& C,
               const arma::mat& Q,
               const arma::mat& R,
               const arma::colvec& F_0,
               const arma::mat& P_0,
               bool retLL) {

  KalmanFilterResult res = SKF_cpp(X, A, C, Q, R, F_0, P_0, retLL);

  return Rcpp::List::create(
    Rcpp::Named("F") = res.F,
    Rcpp::Named("P") = res.P,
    Rcpp::Named("F_pred") = res.F_pred,
    Rcpp::Named("P_pred") = res.P_pred,
    Rcpp::Named("loglik") = res.loglik
  );
}

// R version
// [[Rcpp::export]]
Rcpp::List SKFS(const arma::mat& X,
                const arma::mat& A,
                const arma::mat& C,
                const arma::mat& Q,
                const arma::mat& R,
                const arma::colvec& F_0,
                const arma::mat& P_0,
                bool retLL) {
  KalmanFilterSmootherResult res = SKFS_cpp(X,A,C,Q,R,F_0,P_0,retLL);

  return Rcpp::List::create(
    Rcpp::Named("F") = res.F,
    Rcpp::Named("P") = res.P,
    Rcpp::Named("F_pred") = res.F_pred,
    Rcpp::Named("P_pred") = res.P_pred,
    Rcpp::Named("F_smooth") = res.F_smooth,
    Rcpp::Named("P_smooth") = res.P_smooth,
    Rcpp::Named("Lag_one_cov_smooth") = res.Lag_one_cov_smooth,
    Rcpp::Named("loglik") = res.loglik
  );
}
