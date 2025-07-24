#pragma once
#include <RcppArmadillo.h>

arma::vec FromSymMatrixToVector(const arma::mat& sym_mat);
arma::mat FromVectorToSymMatrix(const arma::vec& sym_vec, int mat_dim);
