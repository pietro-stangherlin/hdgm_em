library(mvtnorm)

# some functions usefull for testing



# State Space Simulation --------------------------------

#' @description Simulate from a linear Gaussian State Space model
#' @param n_times (int): number of observations
#' @param obs_dim (int): dimension of the observed vector
#' @param state_dim (int): dimension of the state vector
#' @param transMatr (matrix): state transition matrix
#' @param obsMatr (matrix): observation transfer matrix
#' @param transMatr (matrix): state transition matrix
#' @param stateCovMatr (matrix): state covariance matrix
#' @param obsCovMatr (matrix): observation error covariance matrix
#' @param zeroState (vector): initial state vector
#' @return (list): with elements: the two matrices of simulated states and obserbations


LinGauStateSpaceSim <- function(n_times = 10^3,
                          obs_dim = 5,
                          state_dim = 3,
                          transMatr = diag(0.5, state_dim),
                          obsMatr = diag(1, state_dim),
                          stateCovMatr = diag(1, state_dim),
                          obsCovMatr = diag(0.1, obs_dim),
                          zeroState = rep(0, obs_dim)){


  X <- matrix(NA, nrow = obs_dim, ncol = n_times) # data matrix
  Z <- matrix(NA, nrow = state_dim, ncol = n_times + 1) # state matrix


  # simulate process
  Z[,1] <- transMatr %*% as.matrix(zeroState) + t(rmvnorm(n = 1,
                                            mean = rep(0, state_dim),
                                            sigma = stateCovMatr))

  for(i in 1:n_times){
    # generate state
    Z[,i+1] <- transMatr %*% as.matrix(Z[,i]) + t(rmvnorm(n = 1,
                                                  mean = rep(0, state_dim),
                                                  sigma = stateCovMatr))
    # generate observation
    X[,i] <- obsMatr %*% as.matrix(Z[,i]) + t(rmvnorm(n = 1,
                                                mean = rep(0, obs_dim),
                                                sigma = obsCovMatr))
  }



  return(list("states" =  Z,
              "observations" = X))

}

# HDGM Simulation ----------------------------------------

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
#' @param sigmay (num): observation error standard deviation
#' @param sigmaz (num): state error standard deviation
#' @param upsilon (num): scaling factor passing from state to observed
#' @param gHDGM (num): state vector autoregressive factor
#' @param z0 (vector): starting state

RHDGM <- function(n,
                  y_len,
                  cor_matr,
                  sigmay,
                  sigmaz,
                  upsilon,
                  gHDGM,
                  z0 = NULL){

  if(is.null(z0)){
    z0 <- rep(0, y_len)
  }

  sigma2z <- sigmaz^2

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
  for(i in 2:(n + 1)){
    z_vals[i,] <- gHDGM * z_vals[i-1,] + rmvnorm(n = 1,
                                                 mean = rep(0, y_len),
                                                 sigma = sigma2z * cor_matr)
    y_vals[i-1,] <- upsilon * z_vals[i,] + rnorm(y_len, sd = sigmay)

  }


  return(list(y = y_vals,
              z = z_vals))

}




