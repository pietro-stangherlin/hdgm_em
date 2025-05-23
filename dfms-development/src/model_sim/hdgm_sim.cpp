#include <armadillo>
#include <random>

#include "model_sim/hdgm_sim.h"

// Compute exponential correlation matrix from distance matrix
arma::mat ExpCor(const arma::mat& mdist, double theta) {
    return arma::exp(-mdist / theta);
}

// Sample from multivariate normal: returns column vector
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& sigma) {
    arma::vec z = arma::randn<arma::vec>(mean.n_elem);
    return mean + arma::chol(sigma) * z;
}


RHDGMResult RHDGM(const int n, // number of observations
                  const int y_len, // observation vector dimension
                  const arma::mat& cor_matr, // spatial state innovation correlation matrix
                  const double sigmay, // observation standard deviation (assuming IID)
                  const double upsilon, // observation matrix scale
                  const double gHDGM, // transition matrix scale (|g| < 1 for identifiability)
                  arma::vec z0) { // initial state

    if (z0.n_elem == 0) {
        z0 = arma::zeros<arma::vec>(y_len);
    }

    arma::mat y_vals(y_len, n, arma::fill::none);
    arma::mat z_vals(y_len, n + 1, arma::fill::none);

    z_vals.col(0) = z0;

    for (int i = 1; i <= n; ++i) {
        arma::vec noise_z = rmvnorm(arma::zeros<arma::vec>(y_len), cor_matr);
        z_vals.col(i) = gHDGM * z_vals.col(i - 1) + noise_z;

        arma::vec noise_y = arma::randn<arma::vec>(y_len) * sigmay;
        y_vals.col(i - 1) = upsilon * z_vals.col(i) + noise_y;
    }

    return {y_vals, z_vals};
}
