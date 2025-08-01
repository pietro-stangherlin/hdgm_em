#pragma once

#include "EM_types.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "EM_algorithm.hpp"
#include "EM_algorithm_impl.hpp"


// [[Rcpp::export]]
Rcpp::List UnstructuredEM( const arma::mat& y, // observation matrix (n x T) where T = n. obs
                          const arma::mat& Phi_0, // initial value transfer matrix
                          const arma::mat& A_0, // initial value observation matrix
                          const arma::mat& Q_0, // initial value state error covariance matrix
                          const arma::mat& R_0, // initial value observation error covariance matrix
                          const arma::vec& x0_in, // initial state
                          const arma::mat& P0_in, // initial state covariance matrix
                          int max_iter,
                          bool bool_mat,
                          bool verbose = true){ // TO change + add tolerance

  EMInputUnstructured em_in{
  .y = y,
  .Phi_0 = Phi_0,
  .A_0 = A_0,
  .Q_0 = Q_0,
  .R_0 = R_0,
  .x0_in = x0_in,
  .P0_in = P0_in,
  .max_iter = max_iter,
  .verbose = verbose
  };

  EMOutputUnstructured res;

  if(bool_mat == true){
    res = UnstructuredEM_cpp_mat(em_in);
  }else{
    res = UnstructuredEM_cpp(em_in);
  };

  return Rcpp::List::create(
    Rcpp::Named("Phi") = res.Phi,
    Rcpp::Named("A") = res.A,
    Rcpp::Named("Q") = res.Q,
    Rcpp::Named("R") = res.R,
    Rcpp::Named("x0_smoothed") = res.x0_smoothed,
    Rcpp::Named("P0_smoothed") = res.P0_smoothed
  );

}



// [[Rcpp::export]]
Rcpp::List EMHDGM(const arma::mat& y, // observation matrix (n x T) where T = n. obs
                    const arma::mat& dist_matrix, // state distance matrix (complete data)
                    double alpha0, // initial observation matrix scaling
                    const arma::vec beta0, // initial fixed effect
                    double theta0, // initial state covariance parameter (exponential)
                    double g0, // initial state transition matrix scaling
                    double sigma20, // initial observations variance
                    bool bool_mat,
                    const arma::vec x0_in,
                    const arma::mat P0_in,
                    const Rcpp::NumericVector& Xbeta_in,
                    const double rel_llik_tol = 1.0e-5, // stopping criterion: relative incremente log likelihood
                    const double theta_lower = 1e-05, // minimum theta value
                    const double theta_upper = 20, // maximum theta value -> this can be incremented but a warning is given
                    int max_iter = 10, // TO change + add tolerance
                    bool is_fixed_effects = false,
                    bool verbose = true) {

  std::cout << "Inside EMHDGM\n";

  // convert array to arma::cube
  arma::cube Xbeta_opt;

  Rcpp::NumericVector Xvec(Xbeta_in);
  Rcpp::IntegerVector dims = Xvec.attr("dim");

  if (dims.size() != 3) {
      Rcpp::stop("Xbeta must be a 3-dimensional array.");
  }

  int n_rows = dims[0];
  int n_cols = dims[1];
  int n_slices = dims[2];

  arma::cube Xbeta(Xvec.begin(), n_rows, n_cols, n_slices, false); // no copy
  Xbeta_opt = Xbeta;

  std::cout << "After optional parameters \n";


  // make input structure
  EMInput inp{
    .y = y,
    .dist_matrix = dist_matrix,
    .alpha0 = alpha0,
    .beta0 = beta0,
    .theta0 = theta0,
    .g0 = g0,
    .sigma20 = sigma20,
    .x0_in = x0_in,
    .P0_in = P0_in,
    .Xbeta = Xbeta_opt,
    .rel_llik_tol = rel_llik_tol,
    .theta_lower = theta_lower,
    .theta_upper = theta_upper,
    .max_iter = max_iter,
    .is_fixed_effect = is_fixed_effects,
    .verbose = verbose,
  };

  EMOutput res;

  std::cout << "After optional EMInput \n";

  if(bool_mat == true){
    res = EMHDGM_cpp_mat(inp);
  }else{
    res = EMHDGM_cpp(inp);
  }

  std::cout << "After EMOutput \n";

  return Rcpp::List::create(
    Rcpp::Named("par_history") = res.par_history,
    Rcpp::Named("beta_history") = res.beta_history,
    Rcpp::Named("llik") = res.llik,
    Rcpp::Named("niter") = res.niter
  );


}



