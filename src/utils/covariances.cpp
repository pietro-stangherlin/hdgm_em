#include <RcppArmadillo.h>
#include "covariances.h"

// Compute exponential correlation matrix from distance matrix
/**
 * @brief Computes an exponential spatial correlation matrix
 *        given a distance matrix and a spatial decay parameter theta.
 *
 * @param mdist Matrix of pairwise distances between spatial locations (p x p)
 * @param theta Spatial decay parameter (positive scalar)
 * @return arma::mat Spatial correlation matrix (p x p)
 */
arma::mat ExpCor(const arma::mat& mdist, double theta) {
  return arma::exp(-mdist / theta);
}


// Make block diagonal matrix given two matrices
// this can be further generalized to more than two matrices...

arma::mat MakeTwoBlockDiag(const arma::mat A, arma::mat B){
  // zeros blocks
  arma::mat ZeroTopRight(A.n_rows, B.n_cols, arma::fill::zeros);
  arma::mat ZeroBottomLeft(B.n_rows, A.n_cols, arma::fill::zeros);

  // join
  arma::mat top = arma::join_rows(A, ZeroTopRight);
  arma::mat low = arma::join_rows(B, ZeroBottomLeft);

  return(arma::join_cols(top, low));
}
