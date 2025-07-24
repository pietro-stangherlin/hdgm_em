#pragma once

#include <RcppArmadillo.h>

arma::vec FromSymMatrixToVector(const arma::mat& sym_mat);
arma::mat FromVectorToSymMatrix(const arma::vec& sym_vec, int mat_dim);


// ----------------- helper ---------------------------//
// Helper: get the t-th covariance matrix
template <typename CovStore>
inline arma::mat GetCov(const CovStore& store, int t, int mat_dim) {
  if constexpr (std::is_same_v<CovStore, arma::cube>) {
    return store.slice(t);
  } else {
    return FromVectorToSymMatrix(store.col(t), mat_dim);
  };
};

