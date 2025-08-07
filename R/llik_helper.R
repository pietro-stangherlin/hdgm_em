library(Rcpp)
library(RcppArmadillo)


Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp")

source("R/model_simulation_helper.R")

#' @description  return the log-likelihood for a HDGM
#' @param param (vector) vector with parameters (a, phi, theta, sigma2y, beta)
#' @param y.matr (matrix) observation vector matrix, each column is an observation
#' @param dist.matrix (matrix) space distance matrix (assuming a fixed order) among all
#' @param X.array (array)
HDGM.Llik <- function(param,
                      y.matr,
                      dist.matr,
                      X.array = NULL){


  a = param[1]
  phi = param[2]
  theta = param[3]
  sigma2y = param[4]

  q = NROW(y.matr)

  # fixed effects
  if(length(param) > 4){
    beta = as.matrix(param[5:length(param)])

    for(t in 1:NCOL(y.matr)){
        index.not.miss <- which(is.finite(y.matr[,t]))
        y.matr[index.not.miss,t] <- y.matr[index.not.miss,t] -
          as.vector(X.array[index.not.miss,,t] %*% beta)
    }
  }

  gc()

  # possible improvement
  # do a first pass of kalman smoother
  # get smoothed initial state and covariance
  # and use them to compute the likelihood
  return(SKF(Y = y.matr,
             Phi = phi * diag(nrow = q),
             A = a * diag(nrow = q),
             Q = ExpCor(mdist = dist.matr,
                        theta = theta),
             R = sigma2y * diag(nrow = q),
             x_0 = rep(0, q),
             P_0 = diag(1, q),
             retLL = TRUE,
             vectorized_cov_matrices = TRUE)$loglik)

}

