// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <optional>


// R version
// [[Rcpp::export]]
Rcpp::List EMHDGM(const arma::mat& y; // observation matrix (n x T) where T = n. obs
                    const arma::mat& dist_matrix; // state distance matrix (complete data)
                    double alpha0; // initial observation matrix scaling
                    const arma::vec beta0; // initial fixed effect
                    double theta0; // initial state covariance parameter (exponential)
                    double g0; // initial state transition matrix scaling
                    double sigma20; // initial observations variance
                    std::optional<arma::cube> Xbeta = std::nullopt;
                    std::optional<arma::vec> z0_in = std::nullopt;
                    std::optional<arma::mat> P0_in = std::nullopt;
                    int max_iter = 10; // TO change + add tolerance
                    double theta_lower = 0.00001; // TO decrease
                    double theta_upper = 10.0; // TO increase
                    bool verbose = true;) {





  // make input structure
  EMInput inp{
    .y = y;
    .dist_matrix = dist_matrix;
    .alpha0 = alpha0;
    .beta0 = beta0;
    .theta0 = theta0;
    .g0 = g0;
    .sigma20 = sigma20;
    .Xbeta = Xbetat;
    .z0_in = z0_in;
    .P0_in = P0_in;
    .max_iter = max_iter;
    .theta_lower = theta_lower;
    .theta_upper = theta_upper;
    .verbose = verbose;
  };



  EMOutput res = EMHDGM_cpp(inp);

  return Rcpp::List::create(
    Rcpp::Named("par_history") = res.par_history,
    Rcpp::Named("beta_history") = res.beta_history
  );


}



