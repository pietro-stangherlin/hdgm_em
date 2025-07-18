rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

source("tests/test_helper.R")

# Simulation 1 -----------------------------------
set.seed(123)

N <- 10000
Y_LEN <- 4
THETA <- 5
G <- 0.8
A <- 3
SIGMAY <- 0.1
SIGMAZ <- 2

# generate x coordinate
# generate y coordinate

POINTS <- matrix(NA,
                 nrow = Y_LEN,
                 ncol = 2)

POINTS[,1] <- runif(Y_LEN)
POINTS[,2] <- runif(Y_LEN)

DIST_MATRIX <- as.matrix(dist(POINTS))
# add noise
for (i in NCOL(DIST_MATRIX)){

}

diag(DIST_MATRIX) = 0

COR_MATRIX <- ExpCor(mdist = DIST_MATRIX,
                     theta = THETA)

ETA_MATRIX <- SIGMAZ^2 * COR_MATRIX # state covariance

y.matr <- LinGauStateSpaceSim(n_times = N,
                            obs_dim = Y_LEN,
                            state_dim = Y_LEN,
                            transMatr = G * diag(nrow = Y_LEN),
                            obsMatr = A * diag(nrow = Y_LEN),
                            stateCovMatr = ETA_MATRIX,
                            obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                            zeroState = rep(0, Y_LEN))$observations


# Testing ---------------------------------

# Starting from true values -------------------------

res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 v0 = SIGMAZ^2,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in = NULL,
                 z0_in = NULL,
                 P0_in = NULL,
                 max_iter = 5, # increment
                 verbose = TRUE)

dim(res_EM$par_history)
colnames(res_EM$par_history)
plot(res_EM$par_history[1,])
plot(res_EM$par_history[2,])
plot(res_EM$par_history[3,])
plot(res_EM$par_history[4,])
plot(res_EM$par_history[5,])







