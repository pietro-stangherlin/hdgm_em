rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/utils/data_handling.cpp",
                rebuild = TRUE)

ArrayToCube(array(rep(1, 9), dim = c(3, 3, 3)))
