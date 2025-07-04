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





