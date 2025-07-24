#include <RcppArmadillo.h>
#include <cmath>

// turn a symmetric matrix element to a vector
// storing only the lower diagonal elements
// Example:
// from
// |A, B, C, D|
// |B, E, F, G|
// |C, F, H, I|
// |D, G, I, L|
// to
// (A, B, C, D, E, F, G, H, I, L)
arma::vec FromSymMatrixToVector(const arma::mat& sym_mat){
  int p = sym_mat.n_cols;
  int vec_len = p * (p + 1) / 2;
  arma::vec res_vec(vec_len);

  int first = 0;
  int last = p - 1;
  for (int i = 0; i < p; ++i){
    res_vec.subvec(first,last) = sym_mat.col(i).subvec(i,p-1);
    first = last + 1;
    last = last + p - 1 - i;
  }

  return res_vec;
}



// turn a vector into a symmetric matrix
// populating the lower diagonal
// Example:
// from
// (A, B, C, D, E, F, G, H, I, L)
// to
// |A, B, C, D|
// |B, E, F, G|
// |C, F, H, I|
// |D, G, I, L|
arma::mat FromVectorToSymMatrix(const arma::vec& sym_vec, int mat_dim){
  int l = sym_vec.n_elem;
  arma::mat res_mat(mat_dim, mat_dim, arma::fill::zeros);

  int index = 0;

  // fill lower + diagonal
  for(int i = 0; i < mat_dim; ++i){
    for(int j = i; j < mat_dim; ++j){
      res_mat(i,j) = sym_vec(index);
      res_mat(j,i) = sym_vec(index);
      ++index;
    }
  }

  return res_mat;
}

