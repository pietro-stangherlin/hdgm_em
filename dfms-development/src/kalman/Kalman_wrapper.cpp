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

  std::cout << "Inside Rcpp wrapper\n";
  std::cout << "X dims: " << X.n_rows << " x " << X.n_cols << std::endl;
  std::cout << "Address of X memory: " << X.memptr() << std::endl;

  KalmanFilterInputByValue inp_val{.X = X,
                        .A = A,
                        .C = C,
                        .Q = Q,
                        .R = R,
                        .F_0 = F_0,
                        .P_0 = P_0,
                        .retLL = retLL};


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
  //
  //
  KalmanFilterResult res = SKF_cpp(inp);

 return Rcpp::List::create(
 Rcpp::Named("F")       = res.F,
 Rcpp::Named("P")       = res.P,
 Rcpp::Named("F_pred")  = res.F_pred,
 Rcpp::Named("P_pred")  = res.P_pred,
 Rcpp::Named("K_last")  = res.K_last,
 Rcpp::Named("C_last")  = res.C_last,
 Rcpp::Named("nc_last") = res.nc_last,
 Rcpp::Named("loglik")  = res.loglik
 );


}

// R version
// [[Rcpp::export]]
Rcpp::List SKFDEBUG(const arma::mat& X,
               const arma::mat& A,
               const arma::mat& C,
               const arma::mat& Q,
               const arma::mat& R,
               const arma::vec& F_0,
               const arma::mat& P_0,
               bool retLL) {


  std::cout << "Inside Rcpp wrapper\n";
  std::cout << "X dims: " << X.n_rows << " x " << X.n_cols << std::endl;
  std::cout << "Address of X memory: " << X.memptr() << std::endl;

  std::cout << "A dims: " << A.n_rows << " x " << A.n_cols << std::endl;
  std::cout << "Address of A memory: " << A.memptr() << std::endl;

  std::cout << "C dims: " << C.n_rows << " x " << C.n_cols << std::endl;
  std::cout << "Address of C memory: " << C.memptr() << std::endl;

  std::cout << "Q dims: " << Q.n_rows << " x " << Q.n_cols << std::endl;
  std::cout << "Address of Q memory: " << Q.memptr() << std::endl;

  std::cout << "R dims: " << R.n_rows << " x " << R.n_cols << std::endl;
  std::cout << "Address of R memory: " << R.memptr() << std::endl;

  std::cout << "F_0 dims: " << F_0.size() << std::endl;
  std::cout << "Address of F_0 memory: " << F_0.memptr() << std::endl;

  std::cout << "P_0 dims: " << R.n_rows << " x " << R.n_cols << std::endl;
  std::cout << "Address of P_0 memory: " << R.memptr() << std::endl;

  // DEBUG Version
  double res_debug = SKFExplicitInput_cpp(X, // observation matrix: each observation is a column
                                          A, // state transition matrix
                                          C, // observation matrix
                                          Q, // state covariance error matrix
                                          R, // observation error covariance matrix
                                          F_0, // initial state
                                          P_0, // initial state covariance
                                          retLL);


}



// R version
// Rcpp::List SKFS(const arma::mat& X,
//                 const arma::mat& A,
//                 const arma::mat& C,
//                 const arma::mat& Q,
//                 const arma::mat& R,
//                 const arma::vec& F_0,
//                 const arma::mat& P_0) {
//
//   // make input struct
//   KalmanFilterInput inp{.X = X,
//                         .A = A,
//                         .C = C,
//                         .Q = Q,
//                         .R = R,
//                         .F_0 = F_0,
//                         .P_0 = P_0};
//
//   KalmanSmootherResult res = SKFS_cpp(inp);
//
//   return Rcpp::List::create(
//     Rcpp::Named("F_smooth") = res.F_smooth,
//     Rcpp::Named("P_smooth") = res.P_smooth,
//     Rcpp::Named("Lag_one_cov_smooth") = res.Lag_one_cov_smooth
//   );
// }
