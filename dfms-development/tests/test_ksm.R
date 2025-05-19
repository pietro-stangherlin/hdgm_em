rm(list = ls())
library(Rcpp)
library(RcppArmadillo)
library(mvtnorm)

Rcpp::sourceCpp("src/helper.cpp")
Rcpp::sourceCpp("src/Kalman_wrapper.cpp")


# generate some data -------------------
N <- 1000 # times: t = 1,..,N
n <- rp <- 2 # y_t dimension
# in this case state dimension = rp = n

X <- matrix(NA, nrow = n, ncol = N) # data matrix
Z <- matrix(NA, nrow = rp, ncol = N + 1) # state matrix
A <- diag(0.8, rp) # transition matrix
C <- diag(1, rp) # observation matrix: diagonal only if rp = n
Q <- diag(1, rp) # state covariance
R <- diag(0.1, n) # Observation covariance (n x n)


F_0 <- rep(0, rp) # Initial state vector (rp x 1)
P_0 <- diag(0.5, rp) # Initial state covariance (rp x rp)

set.seed(123)

# simulate process
Z[,1] <- A %*% as.matrix(F_0) + t(rmvnorm(n = 1,
                             mean = rep(0, rp),
                             sigma = Q))

for(i in 1:N){
  # generate state
  Z[,i+1] <- A %*% as.matrix(Z[,i]) + t(rmvnorm(n = 1,
                                              mean = rep(0, rp),
                                              sigma = Q))
  # generate observation
  X[,i] <- C %*% as.matrix(Z[,i]) + t(rmvnorm(n = 1,
                                              mean = rep(0, n),
                                              sigma = R))
}

# so the transposition operation doesn't affect the benchmark
X.t <- t(X)
# test Kalman Filter and Smoother -------------------

# Filter -----------------------------------------

custom.skf.res <- SKF(X = X, A = A, C = C, Q = Q, R = R,
                F_0 = F_0, P_0 = P_0, retLL = FALSE)

# original library
original.skf.res <- dfms::SKF(X = t(X), A = A, C = C, Q = Q, R = R,
          F_0 = F_0, P_0 = P_0)


# benchmark

CustomFilterBench <- function(b){
  start = Sys.time()

  for(i in 1:b){
    v <- SKF(X = X, A = A, C = C, Q = Q, R = R,
             F_0 = F_0, P_0 = P_0)
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

B = 100

OriginalFilterBench(b = B)
CustomFilterBench(b = B)

# Smoother ----------------------------------------------

custom.skfs.res <- SKFS(X = X, A = A, C = C, Q = Q, R = R,
                      F_0 = F_0, P_0 = P_0)

original.skfs.res <- dfms::SKFS(X = t(X), A = A, C = C, Q = Q, R = R,
                              F_0 = F_0, P_0 = P_0)

# check equivalence
custom.skfs.res$F_smooth[,1:5] == t(original.skfs.res$F_smooth[1:5,])


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

B = 100

OriginalSmootherBench(b = B)
CustomSmootherBench(b = B)













