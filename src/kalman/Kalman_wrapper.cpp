// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Kalman_internal.h"


// R version
// [[Rcpp::export]]
Rcpp::List SKF(const arma::mat& X,
               const arma::mat& A,
               const arma::mat& C,
               const arma::mat& Q,
               const arma::mat& R,
               const arma::vec& F_0,
               const arma::mat& P_0,
               bool retLL) {

  // std::cout << "Inside Rcpp wrapper\n";
  // std::cout << "X dims: " << X.n_rows << " x " << X.n_cols << std::endl;
  // std::cout << "Address of X memory: " << X.memptr() << std::endl;


  // make input struct
  KalmanFilterInput inp{.Y = X,
                        .Phi = A,
                        .A = C,
                        .Q = Q,
                        .R = R,
                        .x_0 = F_0,
                        .P_0 = P_0,
                        .retLL = retLL};

  // std::cout << "AFTER KalmanFilterInput inp \n";
  // std::cout << "Address of X memory: " << inp.X.memptr() << std::endl;
  // std::cout << "X dims: " << inp.X.n_rows << " x " << inp.X.n_cols << std::endl;
  //
  //
  KalmanFilterResult res = SKF_cpp(inp);

 return Rcpp::List::create(
 Rcpp::Named("xf")       = res.xf,
 Rcpp::Named("Pf")       = res.Pf,
 Rcpp::Named("Fp")  = res.xp,
 Rcpp::Named("Pp")  = res.Pp,
 Rcpp::Named("K_last")  = res.K_last,
 Rcpp::Named("A_last")  = res.A_last,
 Rcpp::Named("nc_last") = res.nc_last,
 Rcpp::Named("loglik")  = res.loglik
 );


}


// R version
// [[Rcpp::export]]
Rcpp::List SKFS(const arma::mat& X,
                const arma::mat& A,
                const arma::mat& C,
                const arma::mat& Q,
                const arma::mat& R,
                const arma::vec& F_0,
                const arma::mat& P_0) {

  // make input struct
  KalmanFilterInput inp{.Y = X,
                        .Phi = A,
                        .A = C,
                        .Q = Q,
                        .R = R,
                        .x_0 = F_0,
                        .P_0 = P_0};

  KalmanSmootherResult res = SKFS_cpp(inp);

  return Rcpp::List::create(
    Rcpp::Named("x_smoothed") = res.x_smoothed,
    Rcpp::Named("P_smoothed") = res.P_smoothed,
    Rcpp::Named("Lag_one_cov_smoothed") = res.Lag_one_cov_smoothed,
    Rcpp::Named("x0_smoothed") = res.x0_smoothed,
    Rcpp::Named("P0_smoothed") = res.P0_smoothed
  );
}
