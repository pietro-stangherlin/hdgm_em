#include <RcppArmadillo.h>
#include "../utils/covariances.h"

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
