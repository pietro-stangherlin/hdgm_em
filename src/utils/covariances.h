#pragma once
#include <RcppArmadillo.h>

arma::mat ExpCor(const arma::mat& mdist, double theta);

arma::mat MakeTwoBlockDiag(const arma::mat A, arma::mat B);
