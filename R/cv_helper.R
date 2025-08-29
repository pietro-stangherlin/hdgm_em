Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp",
                rebuild = TRUE)

source("R/model_simulation_helper.R")


# cross-validation helper

# NOTE:
# if there are no tuning parameters
# this is just a way to estimate the error

# NOTE: the predictions are obtained using the kalman filter

# Expanding windows ---------------------------
# consider all the stations
# each time the training data window increases
# in our case 365 seems reasonable because we first have to
# estimate each month effect
# so we let see one full year of data before prediction

#' @param y_matr
#' @param x_array
#' @param dist_matr spatial distant matrix
#' @param starting_obs (int): the number of starting observations
#' @param step_ahead_pred (int): how many steps ahead prediction
#' @param initial_est_structural (vector): initial estimates for structural parameters
#' @param initial_est_fixed (vector): initial estimates for fixed parameters
CVExpandingWindow <- function(y_matr,
                            X_array,
                            dist_matr,
                            initial_est_structural,
                            intial_est_fixed,
                            starting_obs = 400,
                            step_ahead_pred = 3,
                            max_EM_iter = 50){
  n = ncol(y_matr)
  q = nrow(y_matr)
  I = diag(q)
  # allocate error matrix
  # example: n = 10, starting_obs = 5, step_ahead_pred = 2
  # hence we only predict observations 7,8,9,10
  # i.e. 4 predictions = 10 - 5 - 2 + 1
  err_ncol = n - starting_obs - step_ahead_pred + 1
  err_matr <- matrix(NA, nrow = q, ncol = err_ncol)


  # use first observations to get an initial estimate
  res_EM <- EMHDGM(y = y.matr[,1:starting_obs],
                   dist_matrix = dists_matr,
                   alpha0 = initial_est_structural[1],
                   beta0 = intial_est_fixed,
                   theta0 = initial_est_structural[3],
                   g0 = initial_est_structural[2],
                   sigma20 = initial_est_structural[4],
                   Xbeta_in = X_array[,,1:starting_obs],
                   x0_in = rep(0, q),
                   P0_in = diag(1, nrow = q),
                   max_iter = max_EM_iter,
                   verbose = FALSE,
                   bool_mat = TRUE,
                   is_fixed_effects = TRUE)

  temp_struct <- res_EM$par_history[,res_EM$niter]
  temp_fixed <- res_EM$beta_history[,res_EM$niter]

  # at each iteration use the previous parameter estimates as starting points
  # each time point increases by one in time
  err_index <- 0
  for(t in starting_obs:(n - step_ahead_pred)){
    print(paste0("cv iter: ", t, collapse = ""))
    err_index <- err_index + 1

    # expand window and updated parameters

    res_EM <- EMHDGM(y = y.matr[,1:t],
                     dist_matrix = dists_matr,
                     alpha0 = temp_struct[1],
                     beta0 = temp_fixed,
                     theta0 = temp_struct[3],
                     g0 = temp_struct[2],
                     sigma20 = temp_struct[4],
                     Xbeta_in = X_array[,,1:t],
                     x0_in = rep(0, q),
                     P0_in = diag(1, nrow = q),
                     max_iter = max_EM_iter,
                     verbose = FALSE,
                     bool_mat = TRUE,
                     is_fixed_effects = TRUE)

    temp_struct <- res_EM$par_history[,res_EM$niter]
    temp_fixed <- res_EM$beta_history[,res_EM$niter]

    # estimate of fixed effects for the predicted observation

    # run kalman filter to predict the non fixed effects component
    # (just use the predicted state value times the transfer matrix value)
    k = t + step_ahead_pred
    err_temp_fixed <- y.matr[,k] - as.vector(X_array[,,k] %*% temp_fixed)

    # matrix of all fixed effects
    temp_fixed_obs <- matrix(NA, nrow = q, ncol = n)

    for(j in 1:t){
      temp_fixed_obs[,j] <- X_array[,,j] %*% temp_fixed
    }

    # run kalman filter to obtain prediction
    custom.skf.res.mat <- SKF(Y = cbind(y.matr[,1:t], matrix(NaN, nrow = q, ncol = step_ahead_pred)),
                              Phi = temp_struct[2] * I,
                              A = temp_struct[1] * I,
                              Q = ExpCor(mdist = dist_matr, theta = temp_struct[3]),
                              R = temp_struct[4] * I,
                              x_0 = rep(0, q),
                              P_0 = diag(1, nrow = q),
                              retLL = TRUE,
                              vectorized_cov_matrices = TRUE)



    err_matr[,err_index] <- err_temp_fixed - temp_struct[1] * custom.skf.res.mat$xp[,ncol(custom.skf.res.mat$xp)]

  }

  return(err_matr)

}

# leave one station out ---------------------
#' @param validation_station_indexes (vector): vector of integer each specifing the
#' index of each station that will be validated (i.e. fit the model on all the stations
#' expect that one and then evaluate the error on that one)
CVLOSO <- function(y_matr,
                   X_array,
                   dist_matr,
                   initial_est_structural,
                   intial_est_fixed,
                   validation_station_indexes,
                   max_EM_iter = 50){

  n = ncol(y_matr)
  q = nrow(y_matr)
  I = diag(q)

  # allocate error matrix: each column for each validation station
  L = length(validation_station_indexes)
  err_matr <- matrix(NA, n, L)

  temp_struct <- initial_est_structural
  temp_fixed <- intial_est_fixed

  for(i in 1:L){
    print(paste0("iter: ", i, collapse = ""))
    # set to NaN current validation station values
    temp_y_matr <- y_matr
    val_ind <- validation_station_indexes[i]
    temp_y_matr[val_ind,] <- NaN

    # estimate parameters without the considered station
    res_EM <- EMHDGM(y = temp_y_matr,
                     dist_matrix = dists_matr,
                     alpha0 = temp_struct[1],
                     beta0 = temp_fixed,
                     theta0 = temp_struct[3],
                     g0 = temp_struct[2],
                     sigma20 = temp_struct[4],
                     Xbeta_in = X_array,
                     x0_in = rep(0, q),
                     P0_in = diag(1, nrow = q),
                     max_iter = max_EM_iter,
                     verbose = FALSE,
                     bool_mat = TRUE,
                     is_fixed_effects = TRUE)

    temp_struct <- res_EM$par_history[,res_EM$niter]
    temp_fixed <- res_EM$beta_history[,res_EM$niter]

    # matrix of all fixed effects
    temp_fixed_obs <- matrix(NA, nrow = q, ncol = n)

    for(j in 1:n){
      temp_fixed_obs[,j] <- X_array[,,j] %*% temp_fixed
    }

    # predict all the observations (actually states) running the kalman filter
    # first remove the fixed effects

    custom.skf.res.mat <- SKFS(Y = temp_y_matr - temp_fixed_obs,
                              Phi = temp_struct[2] * I,
                              A = temp_struct[1] * I,
                              Q = ExpCor(mdist = dist_matr, theta = temp_struct[3]),
                              R = temp_struct[4] * I,
                              x_0 = rep(0, q),
                              P_0 = diag(1, nrow = q),
                              retLL = TRUE)

    # compute errors by subtracting both fixed and non fixed effects
    err_matr[,i] <- y_matr[val_ind,] - temp_fixed_obs[val_ind,] - custom.skf.res.mat$x_smoothed[val_ind,]

  }

  return(err_matr)

}


# mix --------------------------------------
# evaluate if it makes sense to mix the expanding windows and leave-one station out




