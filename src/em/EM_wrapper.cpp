// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <optional>
#include "EM_algorithm.h"

// R version
// [[Rcpp::export]]
Rcpp::List EMHDGM(const arma::mat& y, // observation matrix (n x T) where T = n. obs
                    const arma::mat& dist_matrix, // state distance matrix (complete data)
                    double alpha0, // initial observation matrix scaling
                    const arma::vec beta0, // initial fixed effect
                    double theta0, // initial state covariance parameter (exponential)
                    double g0, // initial state transition matrix scaling
                    double sigma20, // initial observations variance
                    Rcpp::Nullable<Rcpp::NumericVector> Xbeta_in = R_NilValue,
                    Rcpp::Nullable<arma::vec> z0_in = R_NilValue,
                    Rcpp::Nullable<arma::mat> P0_in = R_NilValue,
                    int max_iter = 10, // TO change + add tolerance
                    double theta_lower = 0.00001, // TO decrease
                    double theta_upper = 10.0, // TO increase
                    bool verbose = true) {

  std::cout << "Inside EMHDGM\n";


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
    .g0 = g0,
    .sigma20 = sigma20,
    .Xbeta = Xbeta_opt,
    .z0_in = z0_opt,
    .P0_in = P0_opt,
    .max_iter = max_iter,
    .theta_lower = theta_lower,
    .theta_upper = theta_upper,
    .verbose = verbose,
  };

  std::cout << "After optional EMInput \n";



  EMOutput res = EMHDGM_cpp(inp);

  std::cout << "After EMOutput \n";

  return Rcpp::List::create(
    Rcpp::Named("par_history") = res.par_history,
    Rcpp::Named("beta_history") = res.beta_history
  );


}



