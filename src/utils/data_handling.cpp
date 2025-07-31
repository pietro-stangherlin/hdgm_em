#pragma once

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::cube ArrayToCube(Rcpp::NumericVector &arr) {
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


// NOT Implemented yet due to too much memory usage
// possible alternatives are storing a vector of consecutive indexes
// and another vector pointing to start and end of each section
/*
 * Given an input matrix of dim r x c, for each column get the indexes of non missing elements.
 * Returns a matrix with dim (r+1) x c.
 * In this new matrix the first k elements of each columns are the (sorted) indexes of the
 * non missing elements in the original matrix while the last row holds the number k - 1
 * i.e the index where to stop reading
 * Example: initial matrix has 5 rows and 20 columns
 * the 4th column has elements (1, NA, NA, 2, NA)
 * the corresponsing column of the returned matrix will be
 * (0,3,0,0,0,1) since the at index 0 we have 1, at index 3 we have two and we stop reading
 * indexes in the new matrix after index 1
 */
