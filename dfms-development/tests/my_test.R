rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/helper.cpp")
Rcpp::sourceCpp("src/KalmanFiltering.cpp")

# test Kalman Filter and Smoother
