#pragma once
#include <RcppArmadillo.h>

arma::vec FromSymMatrixToVector(arma::mat& sym_mat);
arma::mat FromVectorToSymMatrix(arma::vec& sym_vec, int mat_dim);
