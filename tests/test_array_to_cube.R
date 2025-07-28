rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/utils/data_handling.cpp",
                rebuild = TRUE)

test_array = array(rep(1, 9), dim = c(3, 3, 3))
test_array[1,2,1] = NA

ArrayToCube(test_array)
