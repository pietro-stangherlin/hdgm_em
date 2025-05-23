#pragma once
#include <armadillo>


arma::mat ExpCor(const arma::mat& mdist, double theta);

arma::vec rmvnorm(const arma::vec& mean, const arma::mat& sigma);

// Result structure: transposed y and z matrices
struct RHDGMResult {
    arma::mat y_vals;  // size: [y_len x n]
    arma::mat z_vals;  // size: [y_len x (n+1)]
};

RHDGMResult RHDGM(const int n, // number of observations
                  const int y_len, // observation vector dimension
                  const arma::mat& cor_matr, // spatial state innovation correlation matrix
                  const double sigmay, // observation variance (assuming IID)
                  const double upsilon, // observation matrix scale
                  const double gHDGM, // transition matrix scale (|g| < 1 for identifiability)
                  arma::vec z0 = arma::vec());