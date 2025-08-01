rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/utils/data_handling.cpp",
                rebuild = TRUE)

N = 10^4
p = 5

test_array = array(rep(1, N*p), dim = c(p, p, N))
test_array[1,2,1] = NA

ArrayToCube(test_array)
