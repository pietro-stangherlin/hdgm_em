rm(list = ls())
library(mvtnorm)

# Hidden dynamic geostatistical model
# Assuming no large scale effects (i.e not fixed effects)

# assuming
# - space: one dimension (continuous)
# - time: one dimension (discrete)

# fix the time points
# fix the space points


ExpCor <- function(mdist, theta){
  exp(-mdist/theta)
}

# generate latent Z
# for each time consider all the space_locations_1d

# initialize latent variable at time zero
# for each location
# in this model the observed vector
# and the state vector have the same dimension

#' @param n (int): number of simulated observations
#' @param y_len (int): dimension of observed and state vectors
#' @param cor_matr (matrix): matrix of spatial correlations of the
#' state vector (same for each time)
#' @param sigma2y (num): variance scale of observed innovations
#' @param sigma2z (num): variance scale of state innovations
#' @param a (num): scaling factor passing from state to observed
#' @param gHDGM (num): state vector autoregressive factor
#' @param z0 (vector): starting state

RHDGM <- function(n,
                  y_len,
                  cor_matr,
                  sigma2y,
                  sigma2z,
                  a,
                  gHDGM,
                  z0 = NULL){

  if(is.null(z0)){
    z0 <- rep(0, y_len)
  }

  y_vals <- matrix(NA,
                     nrow = n,
                     ncol = y_len)

  z_vals <- matrix(NA,
                     nrow = n + 1,
                     ncol = y_len)
  z_vals[1,] <- z0

  # NOTE: this is time inefficient
  # because we can just simulate from the gaussian once
  # but in this way the data generating process is clear

  sigmay <- sqrt(sigma2y)

  for(i in 2:(n + 1)){
    # NOTE: for the moment the state variables innovations only present
    # unitary variance, this has to be fixed
    z_vals[i,] <- gHDGM * z_vals[i-1,] + rmvnorm(n = 1,
                                                  mean = rep(0, y_len),
                                                  sigma = sigma2z * cor_matr)

    y_vals[i-1,] <- a * z_vals[i,] + rnorm(y_len, sd = sigmay)

  }


  return(list(y = y_vals,
              z = z_vals))

}




