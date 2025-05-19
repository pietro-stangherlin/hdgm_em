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

  std::cout << "Inside Rcpp wrapper\n";
  std::cout << "X dims: " << X.n_rows << " x " << X.n_cols << std::endl;
  std::cout << "Address of X memory: " << X.memptr() << std::endl;


  // make input struct
  KalmanFilterInput inp{.X = X,
                        .A = A,
                        .C = C,
                        .Q = Q,
                        .R = R,
                        .F_0 = F_0,
                        .P_0 = P_0,
                        .retLL = retLL};

  std::cout << "AFTER KalmanFilterInput inp \n";
  std::cout << "Address of X memory: " << inp.X.memptr() << std::endl;
  std::cout << "X dims: " << inp.X.n_rows << " x " << inp.X.n_cols << std::endl;


  KalmanFilterResult res = SKF_cpp(inp);

  return Rcpp::List::create(
    //Rcpp::Named("F")       = res.F,
    //Rcpp::Named("P")       = res.P,
    //Rcpp::Named("F_pred")  = res.F_pred,
    //Rcpp::Named("P_pred")  = res.P_pred,
    //Rcpp::Named("K_last")  = res.K_last,
    //Rcpp::Named("C_last")  = res.C_last,
    //Rcpp::Named("nc_last") = res.nc_last,
    //Rcpp::Named("loglik")  = res.loglik
  );
}

// R version
// [[Rcpp::export]]
Rcpp::List SKFS(const arma::mat X,
                const arma::mat A,
                const arma::mat C,
                const arma::mat Q,
                const arma::mat R,
                const arma::vec F_0,
                const arma::mat P_0) {

  // make input struct
  KalmanFilterInput inp{.X = X,
                        .A = A,
                        .C = C,
                        .Q = Q,
                        .R = R,
                        .F_0 = F_0,
                        .P_0 = P_0};

  KalmanSmootherResult res = SKFS_cpp(inp);

  return Rcpp::List::create(
    Rcpp::Named("F_smooth") = res.F_smooth,
    Rcpp::Named("P_smooth") = res.P_smooth,
    Rcpp::Named("Lag_one_cov_smooth") = res.Lag_one_cov_smooth
  );
}
