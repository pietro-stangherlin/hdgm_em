# following Shumway and Stoffer 6.7

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp")

source("R/model_simulation_helper.R")

#' @param mle (vector) vector with mle parameters (a, phi, theta, sigma2y, beta)
#' @param y.matr (matrix) observation vector matrix, each column is an observation
#' @param dist.matrix (matrix) space distance matrix (assuming a fixed order) among all
#' @param X.array (array)
BootstrapHDGM <- function(mle.structural, mle.beta.fixed, y.matr, dist.matrix, X.array, B){



  a = mle.structural[1]
  phi = mle.structural[2]
  theta = mle.structural[3]
  sigma2y = mle.structural[4]

  boot.structural <- matrix(NA, nrow = length(mle.structural), ncol = B)

  q = NROW(y.matr)

  # fixed effects
  if(!is.null(mle.beta.fixed)){
    beta = as.matrix(mle.beta.fixed)
    boot.beta.fixed <- matrix(NA, nrow = length(beta), ncol = B)

    for(t in 1:NCOL(y.matr)){
      index.not.miss <- which(is.finite(y.matr[,t]))
      y.matr[index.not.miss,t] <- y.matr[index.not.miss,t] -
        as.vector(X.array[index.not.miss,,t] %*% beta)
    }
  }

  # run kalman filter to obtain errors to sample
  errs <- matrix(NA, nrow = NROW(y.matr), ncol = NCOL(y.matr))
  err.Sigmas.chols <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))
  err.Sigmas.chols.inv <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))
  kalman.gains <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))

  mle.A <- a * diag(nrow = q)
  mle.R <- sigma2y * diag(nrow = q)
  mle.Phi <- phi * diag(nrow = q)

  kf.res <- SKF(Y = y.matr,
                Phi = mle.Phi,
                A = mle.A,
                Q = ExpCor(mdist = dist.matr,
                           theta = theta),
                R = mle.R,
                x_0 = rep(0, q),
                P_0 = diag(1, q),
                retLL = TRUE,
                vectorized_cov_matrices = TRUE)

  rm(kf.res)

  for(i in 1:NCOL(y.matr)){
    errs[,i] <- y.matr[,i] - as.vector(mle.A %*% kf.res$xp[,i])
    # NOTE: inefficient both storing the lower triangular both
    # computing inverse without using the triangular property
    temp_matr <- kf.res$Pp[,,i] %*% t(mle.A)
    erss.Sigmas.chols[,,i] <- t(chol(mle.A %*% temp_matr + mle.R))
    erss.Sigmas.chols.inv[,,i] <- solve(erss.Sigmas.chols[,,i])
    kalman.gains[,,i] <- (mle.Phi * temp_matr + mle.Phi)
  }




}
