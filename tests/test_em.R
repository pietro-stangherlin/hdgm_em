rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

source("tests/test_helper.R")

Rcpp::sourceCpp("src/em/EM_wrapper.cpp")

# Simulation 1 -----------------------------------
set.seed(123)

N <- 1000
Y_LEN <- 2
THETA <- 5
G <- 0.8
A <- 3
SIGMAY <- 0.1
SIGMAZ <- 1

DIST_MATRIX <- matrix(c(0, 1,
                        1, 0),
                      ncol = Y_LEN,
                      nrow = Y_LEN)

COR_MATRIX <- ExpCor(mdist = DIST_MATRIX,
                     theta = THETA)

res <- RHDGM(n = N,
             y_len = Y_LEN,
             state_cov_matr = SIGMAZ^2 * COR_MATRIX,
             sigmay = SIGMAY,
             a = A,
             gHDGM = G,
             z0 = rep(0, Y_LEN))

y.matr <- t(res$y)


# Testing ---------------------------------

# Starting from true values -------------------------

res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 v0 = 1,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in = NULL,
                 z0_in = NULL,
                 P0_in = NULL,
                 max_iter = 10,
                 verbose = TRUE)

dim(res_EM$par_history)
colnames(res_EM$par_history)
plot(res_EM$par_history[1,])
plot(res_EM$par_history[2,])
plot(res_EM$par_history[3,])
plot(res_EM$par_history[4,])








