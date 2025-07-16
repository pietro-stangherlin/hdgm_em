library(Rcpp)
library(RcppArmadillo)

rm(list = ls())

Rcpp::sourceCpp("tests/test_nelder_mead_wrapper.cpp")


MyNelderMead(5, 5, 0.1, 0.1)
MyNelderMeadAdditionalArgs(5, 5, 0.1, 0.1, 6)
