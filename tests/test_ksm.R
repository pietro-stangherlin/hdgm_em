rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

source("tests/test_helper.R")

# This testing requires the dfms package

# NOT USED
# Rcpp::sourceCpp("src/helper.cpp")

Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp")


# generate some data -------------------
N <- 1000 # times: t = 1,..,N
n <- rp <- 5 # y_t dimension
# in this case state dimension = rp = n


A <- diag(0.8, rp) # transition matrix
C <- diag(1, rp) # observation matrix: diagonal only if rp = n
Q <- diag(1, rp) # state covariance
R <- diag(0.1, n) # Observation covariance (n x n)


F_0 <- as.vector(rep(0, n))  # Initial state vector (rp x 1)
P_0 <- diag(0.5, rp) # Initial state covariance (rp x rp)

set.seed(123)

sim_res <- LinGauStateSpaceSim(n_times = N,
                                obs_dim = n,
                                state_dim = rp,
                                transMatr = A,
                                obsMatr = C,
                                stateCovMatr = Q,
                                obsCovMatr = R,
                                zeroState = F_0)

X <- sim_res$observations
Z <- sim_res$states

# so the transposition operation doesn't affect the benchmark
X.t <- t(X)
# test Kalman Filter and Smoother -------------------

# Filter -----------------------------------------

# original library
original.skf.res <- dfms::SKF(X = X.t, A = A, C = C, Q = Q, R = R,
                              F_0 = F_0, P_0 = P_0)

custom.skf.res <- SKF(Y = X, Phi = A, A = C, Q = Q, R = R,
                x_0 = F_0, P_0 = P_0, retLL = TRUE)


plot(Z[1,], type = "l",
     xlab = "times",
     ylab = "true states")

lines(original.skf.res$F[,1],
      col = "red")

lines(custom.skf.res$xf[1,],
      col = "blue")

# check equivalence
custom.skf.res$xf[,1:5] == t(original.skf.res$F[1:5,])
# benchmark

CustomFilterBench <- function(b){
  start = Sys.time()

  for(i in 1:b){
    v <- SKF(X = X, A = A, C = C, Q = Q, R = R,
             F_0 = F_0, P_0 = P_0, retLL = FALSE)
  }

  return(Sys.time() - start)
}


OriginalFilterBench <- function(b){
  start = Sys.time()


  for(i in 1:b){
    v <- dfms::SKF(X = X.t, A = A, C = C, Q = Q, R = R,
             F_0 = F_0, P_0 = P_0)
  }

  return(Sys.time() - start)
}

B = 1000

# OriginalFilterBench(b = B)
# CustomFilterBench(b = B)


# Smoother ----------------------------------------------

custom.skfs.res <- SKFS(Y = X, Phi = A, A = C, Q = Q, R = R,
                      x_0 = F_0, P_0 = P_0, retLL = TRUE)

original.skfs.res <- dfms::SKFS(X = t(X), A = A, C = C, Q = Q, R = R,
                              F_0 = F_0, P_0 = P_0)

plot(Z[1,], type = "l",
     xlab = "times",
     ylab = "true states")

lines(original.skf.res$F[,1],
      col = "red")

lines(custom.skf.res$xf[1,],
      col = "blue")

lines(original.skfs.res$F_smooth[,1],
      col = "orange")

lines(custom.skfs.res$xs[1,],
      col = "violet")

# check equivalence
custom.skfs.res$xs[,1:5] == t(original.skfs.res$F_smooth[1:5,])

# this should be different
custom.skf.res$F[,1:5] == custom.skfs.res$F_smooth[,1:5]

CustomSmootherBench <- function(b){
  start = Sys.time()

  for(i in 1:b){
    v <- SKFS(X = X, A = A, C = C, Q = Q, R = R,
             F_0 = F_0, P_0 = P_0)
  }

  return(Sys.time() - start)
}


OriginalSmootherBench <- function(b){
  start = Sys.time()


  for(i in 1:b){
    v <- dfms::SKFS(X = X.t, A = A, C = C, Q = Q, R = R,
                   F_0 = F_0, P_0 = P_0)
  }

  return(Sys.time() - start)
}

B = 1000

# OriginalSmootherBench(b = B)
# CustomSmootherBench(b = B)










