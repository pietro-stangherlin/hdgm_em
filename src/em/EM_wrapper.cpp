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
                          bool bool_mat){ // TO change + add tolerance

  EMInputUnstructured em_in{
  .y = y,
  .Phi_0 = Phi_0,
  .A_0 = A_0,
  .Q_0 = Q_0,
  .R_0 = R_0,
  .x0_in = x0_in,
  .P0_in = P0_in,
  .max_iter = max_iter
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
    Rcpp::Named("R") = res.R
  );

}



// [[Rcpp::export]]
Rcpp::List EMHDGM(const arma::mat& y, // observation matrix (n x T) where T = n. obs
                    const arma::mat& dist_matrix, // state distance matrix (complete data)
                    double alpha0, // initial observation matrix scaling
                    const arma::vec beta0, // initial fixed effect
                    double theta0, // initial state covariance parameter (exponential)
                    double v0, // initial state scale covariance parameter (exponential)
                    double g0, // initial state transition matrix scaling
                    double sigma20, // initial observations variance
                    Rcpp::Nullable<Rcpp::NumericVector> Xbeta_in = R_NilValue,
                    Rcpp::Nullable<arma::vec> z0_in = R_NilValue,
                    Rcpp::Nullable<arma::mat> P0_in = R_NilValue,
                    const double var_terminating_lim = 1.0e-4, // stopping criterion for nelder-mead method: variance of values
                    int max_iter = 10, // TO change + add tolerance
                    bool verbose = true) {

  std::cout << "Inside EMHDGM\n";

  // WARNING: this is a temporary solution
  // TO DO: move this to input, but without changing anything
  // this will give problems not recognizing the function
  const std::array<double,2> theta_v_step = {0.0001, 0.0001};

  // convert array to arma::cube
  std::optional<arma::cube> Xbeta_opt = std::nullopt;

  if (Xbeta_in.isNotNull()) {
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
  }

  std::optional<arma::vec> z0_opt = std::nullopt;
  if (z0_in.isNotNull()) {
    z0_opt = Rcpp::as<arma::vec>(z0_in);
  }

  std::optional<arma::mat> P0_opt = std::nullopt;
  if (P0_in.isNotNull()) {
    P0_opt = Rcpp::as<arma::mat>(P0_in);
  }

  std::cout << "After optional parameters \n";


  // make input structure
  EMInput inp{
    .y = y,
    .dist_matrix = dist_matrix,
    .alpha0 = alpha0,
    .beta0 = beta0,
    .theta0 = theta0,
    .v0 = v0,
    .g0 = g0,
    .sigma20 = sigma20,
    .Xbeta = Xbeta_opt,
    .z0_in = z0_opt,
    .P0_in = P0_opt,
    .theta_v_step = theta_v_step,
    .var_terminating_lim = var_terminating_lim,
    .max_iter = max_iter,
    .verbose = verbose
  };

  std::cout << "After optional EMInput \n";

  EMOutput res = EMHDGM_cpp(inp);

  std::cout << "After EMOutput \n";

  return Rcpp::List::create(
    Rcpp::Named("par_history") = res.par_history,
    Rcpp::Named("beta_history") = res.beta_history
  );


}



