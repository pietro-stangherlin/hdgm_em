# following Shumway and Stoffer 6.7

Sys.setenv("PKG_CXXFLAGS"="-std=c++20")
library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp")
Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

source("R/model_simulation_helper.R")

#' @param mle (vector) vector with mle parameters (a, phi, theta, sigma2y, beta)
#' @param y.matr (matrix) observation vector matrix, each column is an observation
#' @param dist.matrix (matrix) space distance matrix (assuming a fixed order) among all
#' @param X.array (array)
BootstrapHDGM <- function(mle.structural, mle.beta.fixed, y.matr, dist.matr, X.array,
                          zero_state, zero_state_var, max_EM_iter, start_obs_index, B){



  a = mle.structural[1]
  phi = mle.structural[2]
  theta = mle.structural[3]
  sigma2y = mle.structural[4]

  boot.structural <- matrix(NA, nrow = length(mle.structural), ncol = B)

  q = NROW(y.matr)

  is_fixed_effects = FALSE

  # fixed effects
  if(!is.null(X.array)){
    is_fixed_effects = TRUE
    beta = as.matrix(mle.beta.fixed)
    boot.beta.fixed <- matrix(NA, nrow = length(beta), ncol = B)

    for(t in 1:NCOL(y.matr)){
      index.not.miss <- which(is.finite(y.matr[,t]))
      y.matr[index.not.miss,t] <- y.matr[index.not.miss,t] -
        as.vector(X.array[index.not.miss,,t] %*% beta)
    }
  }

  # run kalman filter to obtain errors to sample
  ortho.errs <- matrix(NA, nrow = NROW(y.matr), ncol = NCOL(y.matr))
  err.Sigmas.chols <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))
  err.Sigmas.chols.inv <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))
  kalman.gains <- array(NA, dim = c(NROW(y.matr), NROW(y.matr), NCOL(y.matr)))

  mle.A <- a * diag(nrow = q)
  mle.R <- sigma2y * diag(nrow = q)
  mle.Phi <- phi * diag(nrow = q)
  mle.Q <- ExpCor(mdist = dist.matr,
                  theta = theta)

  kf.res <- SKF(Y = y.matr,
                Phi = mle.Phi,
                A = mle.A,
                Q = mle.Q,
                R = mle.R,
                x_0 = zero_state,
                P_0 = zero_state_var,
                retLL = TRUE,
                vectorized_cov_matrices = FALSE)

  new_zero_state <- kf.res$xf[,start_obs_index]
  new_zero_state_var <- kf.res$Pf[,,start_obs_index]



  for(i in 1:NCOL(y.matr)){
    # NOTE: inefficient both storing the lower triangular both
    # computing inverse without using the triangular property
    temp_matr <- kf.res$Pp[,,i] %*% t(mle.A)
    err.Sigmas.chols[,,i] <- t(chol(mle.A %*% temp_matr + mle.R))
    err.Sigmas.chols.inv[,,i] <- solve(err.Sigmas.chols[,,i])
    ortho.errs[,i] <- err.Sigmas.chols.inv[,,i] %*% (y.matr[,i] - mle.A %*% kf.res$xp[,i])
    # NOTE: in the equation below is missing the term
    # Phi %*% S where S is a matrix related to the correlation between
    # observation and state errors (here assumed S = 0)
    # (6.116) formulas
    # kalman.gains[,,i] <- (mle.Phi * temp_matr) %*% err.Sigmas.chols.inv[,,i] %*% t(err.Sigmas.chols.inv[,,i])
    # using uncorrelated errors kalman gain
    kalman.gains[,,i] <- temp_matr %*% err.Sigmas.chols.inv[,,i] %*% t(err.Sigmas.chols.inv[,,i])

  }

  rm(kf.res)

  print("Bootstrap preprocessing done")

  # actual bootstrap
  for(b in 1:B){
    print(paste0("iter: ", b, collapse = ""))

    temp_ortho_errs <- ortho.errs[,sample(1:NCOL(ortho.errs), replace = TRUE)]
    temp_y_matr <- matrix(NA, nrow = NROW(temp_ortho_errs), ncol = NCOL(temp_ortho_errs))

    x_temp <- zero_state

    # simulate observations
    # NOTE: the book suggest to skip the first observations due to
    # first estimates high variance
    for(i in 1:NCOL(temp_ortho_errs)){
      x_temp <- mle.Phi %*% x_temp + kalman.gains[,,i] %*% err.Sigmas.chols[,,i] %*% ortho.errs[,i]
      temp_y_matr[,i] <- as.vector(mle.A %*% x_temp +  err.Sigmas.chols[,,i] %*% ortho.errs[,i])
    }

    # run EM
    res_EM <- EMHDGM(y = temp_y_matr[,start_obs_index:NCOL(temp_y_matr)],
                     dist_matrix = dist.matr,
                     alpha0 = a,
                     beta0 = mle.beta.fixed, # start with OLS estimate
                     theta0 = theta,
                     g0 = phi,
                     sigma20 = sigma2y,
                     Xbeta_in = X.array[,,start_obs_index:NCOL(temp_y_matr)],
                     x0_in = new_zero_state,
                     P0_in = new_zero_state_var,
                     max_iter = max_EM_iter,
                     verbose = FALSE,
                     bool_mat = TRUE,
                     is_fixed_effects = is_fixed_effects)

    boot.structural[,b] <- res_EM$par_history[,res_EM$niter]
    if(is_fixed_effects){
      boot.beta.fixed[,b] <- res_EM$beta_history[,res_EM$niter]
    }

  }


  if(is_fixed_effects){
    return(list("struct" = boot.structural,
                "fixed" = boot.beta.fixed))
  }

  return(list("struct" = boot.structural))

}
