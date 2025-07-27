#pragma once

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::cube ArrayToCube(Rcpp::NumericVector arr) {
  Rcpp::IntegerVector dims = arr.attr("dim");
  if (dims.size() != 3) {
    Rcpp::stop("Input must be a 3D array.");
  }

  int n_rows = dims[0];
  int n_cols = dims[1];
  int n_slices = dims[2];

  arma::cube result(arr.begin(), n_rows, n_cols, n_slices, false); // false = no copy

  return result;
}
