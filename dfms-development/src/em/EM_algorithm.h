#include<armadillo>
#include<optional>

struct EMInput{
  const arma::mat& y; // observation matrix (n x T) where T = n. obs
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
  double sigma2_lower = 0.00001; // TO decrease
  double sigma2_upper = 10.0; // TO increase
  bool verbose = true;
};


struct EMOutput{
  arma::mat par_history; // matrix (k x iter) each column is iter value of (alpha,theta,g, sigma2)^T
  arma::mat beta_history; // (p x iter) each column is a beta (fixed effect value)
};

// change return type
EMOutput EMHDGM(EMInput);
