rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_functions.cpp")
Rcpp::sourceCpp("src/em/EM_functions.cpp")
Rcpp::sourceCpp("src/em/EM_algorithm.cpp")


# Constants -------------------------------
# T = 3
# n = rp = 2

Y <- matrix(c(1,1,1,
              1,1,1),nrow = 2, ncol = 3)

Z <- Y # smoothed states
Xz <- diag(0.5, 2)
Ps <- array(NA, dim = c(2,2,3)) # smoothed variances
Ps[,,1] <- diag(1, 2)
Ps[,,2] <- diag(1, 2)
Ps[,,3] <- diag(1, 2)

# AlphaUpdate ----------------------------------------------

AlphaUpdate(mY_fixed_res = Y,
            mZ = Z,
            mXz = Xz,
            cPsm = Ps)












