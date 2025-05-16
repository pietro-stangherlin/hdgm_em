# Calculli et. al. 2015 HDGM EM implementation
# for the moment I'll only consider a univariate response
# (observed at different locations)
# so, in paper notation q = 1.

# the reference is Paragraph 3.2
# and the Appendix

require(MARSS)


# covariance specification ----------------------------
# where d is a distance matrix
# theta > 0
ExpCor <- function(mdist, theta){
  exp(-mdist/theta)
}

# Utils ---------------------------------------------

# Permutation matrix D
# this map the partitioned \tiled{y}_t = (y^1_t, y^2_t) into the original y_t
# where y^1_t is the vector of observed values at time t
# and y^2_t is the vector of unobserved values at time t

# maybe better sparse since is made mostly by zeros
#' @param vobs_indexes (vector): vector of non missing observations indexes
#' @param y_len (int): length of y vector if no missing value is found 
#' @return sparse matrix
DPermutMatrix <- function(vobs_indexes, y_len){
  return(Matrix::sparseMatrix(i = vobs_indexes,
                              j = vobs_indexes,
                              dims = c(y_len, y_len), x = 1))
}

# EM Updating equations -----------------------------

# it is assumed that, for a fixed set of S spatial locations
# at each time t we can observe a subset of those locations.
# so the input response is a matrix with S rows and T columns
# where some row can be NA in case of missing observation at that time
# for that location.

# similarly, for each time t is considered the fixed effects
# covariates matrix X_t (S x p) which also has rows with NA associated with the
# corresponding NAs in the response matrix
# an array X = (X_1, X_2,.., X_T) is saved where the third index goes from 1 to T

# finally a list of T elements is saved where each element is the index of
# non missing response values which will be used to extract the NOT NA from
# both the response and the fixed effects model matrix

# NOTE: maybe this is not the most efficient implementation since
# we're allocating memory to store NA values, 
# an alternative implementation can make use of list to store 
# only not NA values
# in order to keep more clairity, for the moment we use matrix / array implementation

# NOTE: sometimes vector are stored as matrices

# here the mapping matrix Q is missing since we consider just one response

#' @description EM update for scale parameter of influence of state variables
#' one observation vector (eq. 4)
#' (common to each state)
#'
#' @param y (matrix): (q x T) matrix of observed vector, NA allowed 
#' (a sorting is assumed, example by spatial locations)
#' @param z (matrix): (s x T) matrix of smoothed state vectors
#' @param Xbeta (array) (q x p x T) array of fixed effects covariates matrices,
#'  each of those is accessed by Xbeta[,t]
#' @param beta (matrix) (p x 1) matrix (i.e. a vector) of fixed effects coef,
#' does NOT change with time
#' @param Xz (array) (s x s) non scaled transfer matrix (assumed constant in time
#' (the complete transfer matrix is scaled by alpha)
#' @param P (array): (s x s x T) array of smoothed state variance matrices,
#'  each of those is accessed by P[,t]
#' @param lnmi (list): list of non missing observation indexes at each time
#'
#' @return (num)
AlphaUpdate <- function(y,
                        z,
                        Xbeta = NULL, beta,
                        Xz, P,
                        lnmi){
  
  num <- 0 # numerator
  den <- 0 # denominator
  
  
  
  for (t in 1:NCOL(y)){
    
      
    sub_term = 0
    
    if(!is.null(Xbeta)){
      sub_term <-  Xbeta[lnmi[[t]],,t] %*% beta
    }

    
    num = num + sum(diag(as.matrix(y[lnmi[[t]],t] - sub_term) %*%
                           t(Xz[lnmi[[t]],] %*% as.matrix(z[, t]))))
    
    den = den + sum(diag(Xz[lnmi[[t]],] %*% 
                           ((as.matrix(z[lnmi[[t]], t]) %*% t(z[lnmi[[t]], t])) +
                                  P[lnmi[[t]],,t]) %*% 
                           t(Xz[lnmi[[t]],]))) 
    
  }
  
  return(num / den)
  
}


#' @description EM update for fixed effect coef (eq. 5)
#'
#' @param y (matrix): (q x T) matrix of observed vector, NA allowed 
#' (a sorting is assumed, example by spatial locations)
#' @param z (matrix): (s x T) matrix of smoothed state vectors
#' @param alpha (num): scaling factor of the state vector on the observed vector
#' in the HDGM model this is the upsilon parameter
#' @param Xbeta (array) (q x p x T) array of fixed effects covariates matrices,
#'  each of those is accessed by Xbeta[,t]
#' @param Xz (array) (s x s) non scaled transfer matr(ix, assumed constant in time
#' (the complete transfer matrix is scaled by alpha)
#' @param beta (matrix) (p x 1) matrix (i.e. a vector) of fixed effects coef,
#' does NOT change with time
#' @param inv_mXbeta_sum (matrix) (p x p) inverse of the sum of fixed effect model
#' matrices cross product relative with only not NA observed vector rows:
#' solve(sum(for (t in 1:T){t(Xbeta[lnmi[[t]],,t]) %*% Xbeta[lnmi[[t]],,t]})).
#' this can be computed only once
#' @param lnmi (list): list of non missing observation indexes at each time
#'
#' @return (matrix): (p x 1) matrix form of the beta vector
BetaUpdate <- function(Xbeta, y,
                       z, alpha,
                       Xz,
                       inv_mXbeta_sum,
                       lnmi){
  
  # here the right multiplication factor is computed
  right_term <- as.matrix(rep(0, dim(Xbeta)[2]))
  
  for(t in 1:NCOL(y)){
    right_term = right_term + t(Xbeta[lnmi[[t]],,t]) %*%
      (y[lnmi[[t]],t] - alpha * (Xz[lnmi[[t]],] %*% z[, t]))
  }
  
  print("[DEBUG](inside beta update function)")
  print("right_term vector:")
  print(right_term)
  
  return(inv_mXbeta_sum %*% right_term)
}


  
# Now defining matrix Omega_one_t (A.3)
# NOTE: both in this equation and in the AlphaUpdate some terms can be computed
# just once (residuals and residuals product)
# for the moment we keep all explicit for clarity


#' @description Omega_one_t definition (A.3)
#'
#' all vector and matrices are taken only in the non missing rows
#' @param yt (matrix): (q x 1) matrix (i.e. vector) of observed vector at time t, NA allowed 
#' (a sorting is assumed, example by spatial locations)
#' @param zt (matrix): (s x 1) matrix (i.e. vector) of smoothed state vector at time t
#' @param alpha (num): scaling factor of the state vector on the observed vector
#' in the HDGM model this is the upsilon parameter
#' @param Xbetat (matrix) (q x p) matrix of fixed effects covariates at time t
#' @param beta (matrix) (p x 1) matrix (i.e. a vector) of fixed effects coef,
#' does NOT change with time
#' @param inv_mXbeta_sum (matrix) (p x p) inverse of the sum of fixed effect model
#' matrices cross product relative with only not NA observed vector rows:
#' solve(sum(for (t in 1:T){t(Xbeta[lnmi[[t]],,]) %*% Xbeta[lnmi[[t]],,]})).
#' this can be computed only once
#' @param Xzt (matrix) (s x s) unscaled transfer matrix at time t
#' @param Pt (matrix): (s x s) matrix of smoothed state variance at time t,
#'
#' @return (matrix): (vnmi x vnmi) matrix

Omega_one_t <- function(yt,
                        Xbetat = NULL,
                        beta,
                        zt,
                        Xzt,
                        vnmi,
                        alpha,
                        Pt){
  
  res <- yt - alpha * Xzt %*% as.matrix(zt)
  
  if(!is.null(Xbetat)){
    res <- res - Xbetat %*% beta
  }
  
  prod_matrix <- alpha * Xzt %*% zt
  
  add_matrix <- alpha^2 * Xzt %*% Pt %*% t(Xzt)
  
  return (res %*% t(res) +
            add_matrix)
  
}


# NOTE: this is memory inefficient
# just to test the all code

#' @description Omega_t definition (following A.2)
#'
#' @param Omega_one_t (matrix): (q x 1) matrix (i.e. vector): first element of the block diag
#' @param zlen (int): length of the state vector
#' @param sigma2 (num): y variance factor
#' @param Dpermt (matrix): (q x q) matrices of permutation matrices at time t to map
# the non - missing vector elements back to a vector of non missing elements
#' @param vnmi (vector): vector of non missing observation indexes at time t
#'
#' @return (vector): (q x 1) only diagonal is returned since it's the only 
# information used is other steps
Omega_t <- function(Omega_one_t,
                    sigma2,
                    ylen,
                    vnmi,
                    Dperm){
  # temp matrix to debug
  inside_prod <- Matrix::bdiag(Omega_one_t, diag(sigma2, ylen - length(vnmi)))
  
  
  res <- as.matrix(Dperm) %*%
    inside_prod %*%
    t(as.matrix(Dperm))
  
  # for now manually extract diagonal
  # beacuse we get an error is using diag():
  # Error in diag(lOmega[[t]]) :
  # long vectors not supported yet: array.c:2216
  
  res_diag <- rep(NA, NCOL(res))
  for(i in 1:NCOL(res)){
    res_diag[i] <- res[i,i]
  }
  
  
  return(res_diag)
}


#' @description EM update for the sigma2 parameter (variance of observed vector)
#' under the assumption of i.i.d conditional to the state variable
#' 
#' @param lOmega (matrix): list of T elements
#' each one of them is a vector with the
#' diagonal elements of Omega_t
#' @param nlen_y (int): length of complete observed vector
#'  (assuming no missing values)
#'  
#' @return (num)
Sigma2Update <- function(lOmega, nlen_y){
  temp_trace <- 0
  
  for(t in 1:length(lOmega)){
    temp_trace = temp_trace + sum(lOmega[[t]])
  }
  
  return(temp_trace / ((length(lOmega)) * nlen_y))
}

#' @description compute the matrix S00 
#' @param smoothed_states (matrix): matrix of smoothed states
#' @param smoothed_vars (array): array of smoothed sates variance matrices
#' @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
#' @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
#' of the nondiffuse part of the initial state vector.
#' @return (matrix) m x m
ComputeS00 <- function(smoothed_states,
                       smoothed_vars,
                       z0,
                       P0){
  S00 <- z0 %*% t(z0) + P0
  
  # all except the last time T
  for(t in 1:(NCOL(smoothed_states) - 1)){
    S00 = S00 + smoothed_states[,t] %*% t(smoothed_states[,t]) + smoothed_vars[,,t]
  }
  
  return(S00)
}

#' @description compute the matrix S11 
#' @param smoothed_states (matrix): matrix of smoothed states
#' @param smoothed_vars (array): array of smoothed sates variance matrices
#' @param S00 (matrix) m x m as defined in the paper
#' @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
#' @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
#' of the nondiffuse part of the initial state vector.
#' @return (matrix) m x m
ComputeS11 <- function(smoothed_states,
                       smoothed_vars,
                       S00,
                       z0,
                       P0){
  # this should be T, (index of last time)
  # but we call it N to avoid confusion with reserved names
  N <- NCOL(smoothed_states)
  
  return(S00 - z0 %*% t(z0) - P0 + 
           smoothed_states[,N] %*% t(smoothed_states[,N]) + smoothed_vars[,,N])
}

#' @description compute the matrix S11 
#' @param smoothed_states (matrix): matrix of smoothed states
#' @param lagone_smoothed_covars (array): array of lag one 
#' smoothed sates covariance matrices
#' @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
#' @return (matrix) m x m
ComputeS10 <- function(smoothed_states,
                       lagone_smoothed_covars,
                       z0){
  S10 <- smoothed_states[,1] %*% t(z0) + lagone_smoothed_covars[,,1]
  
  for(t in 2:NCOL(smoothed_states)){
    S10 <- S10 + smoothed_states[,t] %*% t(smoothed_states[,t-1]) + lagone_smoothed_covars[,,t]
  }
  
  return(S10)
  
}

# NOTE: I need to find a way to find P_{t,t-1}
# this is the smoothed lag-one
# NOT computed by KFAS, but computed in a slower way
# by MARSS using MARSSkfas and using a state vector
# of double the size
# see details (return.lag.one = TRUE)
?MARSSkfas
MARSSkfas

# see: R. H. Shumway and D. S. Stoffer (2006). 
# Time series analysis and its applications: with R examples
# Property 6.3: The Lag-One Covariance Smoother

# eq. (7)
# where S10 and S11 are defined in the article
# and are function of the (Kalman) smoothed vectors
#' @description EM update for the g constant, this is g_HDGM in the HDGM paper
#' i.e. the autoregressive coefficient
#' @param S00 (matrix): given the kalman smoothed states
#' matrix_sum(for(t in 1:T)({z_{t-1} %*% t(z_{t-1}) + P_{t-1}}) and
#' P_{t-1} is the smoothed variance at time t-1
#' @param S10 (matrix): given the kalman smoothed states
#' matrix_sum(for(t in 1:T)({z_{t} %*% t(z_{t-1}) + P_{t,t-1}}) and
#' P_{t,t-1} is the smoothed covariance between times t and t-1
#' 
#' @return (num)
gUpdate <- function(S00, S10){
  sum(diag(S10)) / sum(diag(S00))
}

# brutal optimization of parameters
# it has to be optimized
# here theta is the theta in exp(-||s-s'|| / theta) of the spatial correlation function
# ng is g_HDGM
# theta0 and g0 starting values

# eq. 9
# where S10 and S11 are defined in the article
# and are function of the (Kalman) smoothed vectors
#' @description EM update for the g constant, this is g_HDGM in the HDGM paper
#' i.e. the autoregressive coefficient
#' @param dist_matrix (matrix): p x p distance matrices between all
#' the state elements, this is used to compute the spatial correlation
#' @param g (num): autoregressive scaling parameter for the state vector
#' hidden process, this is g_HDGM in the HDGM paper
#' @param theta0 (num): starting value inverse weight in the exponential spatial correlation
#' for the state vector error (theta_HDGM in the HDGM paper)
#' @param S00 (matrix): given the kalman smoothed states
#' matrix_sum(for(t in 1:T)({z_{t-1} %*% t(z_{t-1}) + P_{t-1}}) and
#' P_{t-1} is the smoothed variance at time t-1
#' @param S10 (matrix): given the kalman smoothed states
#' matrix_sum(for(t in 1:T)({z_{t} %*% t(z_{t-1}) + P_{t,t-1}}) and
#' P_{t,t-1} is the smoothed covariance between times t and t-1
#' @param S11 (matrix): given the kalman smoothed states
#' matrix_sum(for(t in 1:T)({z_{t} %*% t(z_{t}) + P_{t}}) and
#' P_{t-1} is the smoothed variance at time t-1,
#' i.e. S11 = S00 - z_0 %*% t(z_0) - P_0 + z_1 %*% t(z_1) + P_1
#' @param N (int): number of times (i.e. number of observations)
#' @param upper (num): maximum value of the parameter
#' @param do_plot_objective (bool): boolean, plot the objective function along
#' with the current estimate
#' 
#' @return (num)

ThetaUpdate <- function(dist_matrix,
                        g,
                        S00,
                        S10,
                        S11,
                        theta0,
                        N,
                        upper = 10,
                        do_plot_objective = TRUE
                        ){
  
  # negative of the function to be maximized
  # NOTE: this is a specific function (exponential spatial correlation)
  # can be generalized

  
  # NOTE 1): I think in the article the sign of the 
  # function to be maximized is wrong.
  # here it is changed.
  
  
  # NOTE 2): this should be log - parameterized 
  # in order to automatically satisfy the non - negative
  # constraint, at the moment it is not implemented
  # for debug reasons
  negative_to_optim <- function(theta){
    
    Sigma_eta <- ExpCor(mdist = dist_matrix,
                        theta = theta)
    return(
    N * determinant(x = Sigma_eta, logarithm = TRUE)$modulus +
      sum(diag(solve(Sigma_eta) %*%
                 (S11 - g * S10 - g * t(S10)  + g^2 * S00))))
  }
  
  negative_to_optim = Vectorize(negative_to_optim,
                                vectorize.args = "theta")
  
  res <- optimize(f = negative_to_optim,
                  lower = 0, upper = upper)$minimum
  
  # diagnostic plot
  if(do_plot_objective){
    curve(negative_to_optim,
          xname = "theta",
          from = 0, to = upper)
    
    abline(v = res, col = "red")
  }
  
  return(res)
  
}



# EM ------------------------------------------

# NOTE: here at each iteration k of the Kalman smoother
# I set the observed matrix as y - X %*% beta_(k-1)

#' @param y (matrix): q x T matrix containing the observations.
#' each column contains an observation vector
#' @param Xz (matrix): q x q of state transition matrix, assumed to be
#' constant in time and made by known terms, it will be scaled by unknown 
#' parameter g0
#' @param dist_matrix (matrix): p x p distance matrices between all
#' the state elements, this is used to compute the spatial correlation
#' @param alpha0 (num): starting value scaling factor of the state vector on the observed vector
#' in the HDGM model this is the upsilon parameter
#' @param beta0 (matrix): starting value this is the starting fixed effects coef,
#' stored as a matrix only for convenience
#' @param g0 (num): starting value autoregressive scaling parameter for the state vector
#' hidden process, this is g_HDGM in the HDGM paper
#' @param theta0 (num): starting value inverse weight in the exponential spatial correlation
#' for the state vector error (theta_HDGM in the HDGM paper)
#' @param sigma20 (num): starting value > 0, starting parameters for the observed vector
#' variance scale (assuming it is diagonal)
#' @param Xbeta (array): q x p x T of fixed effects covariates
#' in HDGM model q = p, if = NULL, no beta update is done
#' (no fixed effects)
#' @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
#' @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
# of the nondiffuse part of the initial state vector.
#' @param max_iter (int): max number of EM iteration
#' (to add tolerance param)
#' @param sigma2_upper (num): maximum value of sigma2 value in the optimization procedure
#' @param sigma2_do_plot_objective (bool): if TRUE plot the objective function used to optimize
#' the value of sigma2 at each iteration, along with the optimized value
EMHDGM <- function(y,
                   dist_matrix,
                   alpha0, 
                   beta0,
                   theta0, 
                   g0,
                   sigma20,
                   Xbeta = NULL,
                   z0 = NULL, 
                   P0 = NULL,
                   max_iter = 10,
                   sigma2_upper = 10,
                   sigma2_do_plot_objective = TRUE){
  
  # first implementation
  is_fixed_effect = !is.null(Xbeta)
  
  beta_temp <- beta0
  
  # debug
  if(is_fixed_effect){
    beta_iter_history <- matrix(NA, nrow = max_iter + 1, ncol = length(beta0))
    beta_iter_history[1,] <- beta0
  }
  
  alpha_temp <- alpha0
  theta_temp <- theta0
  g_temp <- g0
  sigma2_temp <- sigma20
  
  z_temp <- z0
  P_temp <- P0
  
  # DEBUG
  # track all parameters
  # except beta
  # each row holds an iteration
  # each column a parameter
  param_history_matrix <- matrix(NA,
                                 nrow = max_iter + 1,
                                 ncol = 4)
  
  colnames(param_history_matrix) = c("alpha", "theta",
                                     "g", "sigma2")
  
  param_history_matrix[1,1] <- alpha_temp
  param_history_matrix[1,2] <- theta_temp
  param_history_matrix[1,3] <- g_temp
  param_history_matrix[1,4] <- sigma2_temp
  
  
  # populated inside the while cycle in the EM step
  lOmega <- list()
  
  q <- NROW(y)
  
  p <- NROW(beta0)
  
  # Xz = transfer matrix
  # Tz = transition matrix
  Xz <- Tz <- diag(1, q)
  
  if(is.null(z0)){
    z_temp <- as.matrix(rep(0, q))
  }
  
  if(is.null(P0)){
    P_temp <- diag(1, q)
  }
  
  # get list of not NA rows
  # assuming the y are already sorted
  # and store Na permutation matrices
  not_na_indexes_list = list()
  not_na_perm_matrices_list = list()
  for(t in 1:NCOL(y)){
    not_na_indexes_list[[t]] <- which(!is.na(y[,t]))
    not_na_perm_matrices_list[[t]] <- DPermutMatrix(vobs_indexes = not_na_indexes_list[[t]],
                                                    y_len = q)
  }
  
  # invert fixed matrix sum once (eq. 5, right matrix factor)
  
  if(is_fixed_effect){
    mXbeta_sum = matrix(0, nrow = p, ncol = p)
    for(t in 1:NCOL(y)){
      mXbeta_sum = mXbeta_sum + t(Xbeta[not_na_indexes_list[[t]],,t]) %*%
        Xbeta[not_na_indexes_list[[t]],,t]
    }
    
    print("[DEBUG]: mXbeta_sum")
    print(mXbeta_sum)
    
    m_inv_mXbeta_sum = solve(mXbeta_sum)
    print("[DEBUG]: m_inv_mXbeta_sum")
    print(m_inv_mXbeta_sum)
  }
  
  
  iter = 0
  # NOTE: Add tolerance stopping criterion
  while(iter < max_iter){
    
    print(paste0(c("iteration", iter), collapse = " "))
    
    #///////////////////////////////////////////
    # KalmanSmoother using updated parameters // ----------
    #//////////////////////////////////////////
    
    print("preprocess phase")
    
    y_res <- y
    
    # get the initialization of fixed effects intercept
    # define new observations subtracting the fixed effects
    # i.e. run KF on residuals
    if(is_fixed_effect){
      
      # this is a q x T of fixed effect intercepts
      u <- matrix(NA, nrow = NROW(y), ncol = NCOL(y))
      
      for (t in 1:dim(Xbeta)[3]){
        u[not_na_indexes_list[[t]],t] <- Xbeta[not_na_indexes_list[[t]],,t] %*% beta_temp
      }
      y_res <- y - u
    }
    
    
    # spatial correlation 
    Q_temp <- ExpCor(mdist = dist_matrix,
                     theta = theta_temp)
    
    # using MARSS to fit the model and then run KFAS
    # smoother on that to get the lag - one smoothed states covariance
    # needed for EM
    # NOTE: MARSS will try to do EM itself
    # but since we give it all the parameters as numeric
    # it will just return a MARSSmodel object
    # to which the smoother will be run
    
    # first KS pass
    print("MARSS model definition")
    
    marss.fit <- MARSS(y_res, # transposition is memory inefficient, to change
                       model = list(B = diag(g_temp, p), # state transition matrix
                                    Z = alpha_temp * Xz, # transfer matrix, assuming p = q
                                    A = "zero", # y time varying intercept 
                                    Q = Q_temp, # state error variance
                                    R = diag(sigma2_temp, q), # observation variance
                                    U = "zero", # fixed effects intercept
                                    x0 = z_temp, # initial state
                                    V0 = P_temp)) # initial state variance
    
    print("Kalman smoother pass start")
    marss.kfas.res <- MARSSkfas(marss.fit,
                                return.lag.one = TRUE,
                                return.kfas.model = FALSE)
    print("Kalman smoother pass finish")
    
    z_smooth_temp <- marss.kfas.res$xtT
    z_smooth_var_temp <- marss.kfas.res$VtT
    
    
    # get smoother output
    S00_temp <- ComputeS00(smoothed_states = z_smooth_temp,
                           smoothed_vars = z_smooth_var_temp,
                           z0 = marss.kfas.res$x0T,
                           P0 = marss.kfas.res$V0T)
    
    S11_temp <- ComputeS11(smoothed_states = z_smooth_temp,
                           smoothed_vars = z_smooth_var_temp,
                           S00 = S00_temp,
                           z0 = marss.kfas.res$x0T,
                           P0 = marss.kfas.res$V0T)
    
    S10_temp <- ComputeS10(smoothed_states = z_smooth_temp,
                           lagone_smoothed_covars = marss.kfas.res$Vtt1T,
                           z0 = marss.kfas.res$x0T)
    
    #////////////////////////////////////////
    # EM update //////////////////////////// ------------------
    #//////////////////////////////////////
    
    print("EM starting")
    
    
    print("Omega build")
    
    for(t in 1:NCOL(y)){
      
      temp_Xbeta_t <- NULL
      
      if(is_fixed_effect){
        temp_Xbeta_t <- Xbeta[not_na_indexes_list[[t]],,t]
      }
      
      
      lOmega[[t]] <- Omega_t(Omega_one_t = Omega_one_t(yt = y[not_na_indexes_list[[t]],t],
                                                       Xbetat = temp_Xbeta_t,
                                                       beta = beta_temp,
                                                       zt = z_smooth_temp[not_na_indexes_list[[t]],t],
                                                       Xzt = Xz[not_na_indexes_list[[t]],],
                                                       vnmi = not_na_indexes_list,
                                                       alpha = alpha_temp,
                                                       Pt = z_smooth_var_temp[not_na_indexes_list[[t]],,t]),
                             sigma2 = sigma2_temp,
                             ylen = q,
                             vnmi = not_na_indexes_list[[t]],
                             Dperm = not_na_perm_matrices_list[[t]])
    }
    
    
    print("Sigma2Update")
    sigma2_temp <- Sigma2Update(lOmega = lOmega,
                                nlen_y = q)
    
    
    print("AlphaUpdate")
    alpha_temp <- AlphaUpdate(y = y,
                              z = z_smooth_temp,
                              Xbeta = Xbeta,
                              beta = beta_temp,
                              Xz = Xz,
                              P = z_smooth_var_temp,
                              lnmi = not_na_indexes_list)
    
    
    if(is_fixed_effect){
      print("BetaUpdate")
      beta_temp <- BetaUpdate(inv_mXbeta_sum =  m_inv_mXbeta_sum,
                            Xbeta = Xbeta,
                            Xz = Xz,
                            y = y,
                            z = z_smooth_temp,
                            alpha = alpha_temp,
                            lnmi = not_na_indexes_list)
    
      beta_iter_history[iter + 1,] <- beta_temp
      print("[DEBUG]: beta_temp")
      print(beta_temp)
    
    }
    
    # NOTE: this optimization should require a reparam.
    # in order to satisfy the constraints
    # here the specific exponential spatial correlation is assumed
    # the function can be made more general
    print("ThetaUpdate")
    theta_temp <- ThetaUpdate(dist_matrix = dist_matrix,
                              g = g_temp,
                              S00 = S00_temp,
                              S10 = S10_temp,
                              S11 = S11_temp,
                              theta0 = theta_temp,
                              N = NCOL(y),
                              upper = sigma2_upper,
                              do_plot_objective = sigma2_do_plot_objective)
    
    print("gUpdate")
    g_temp <- gUpdate(S00 = S00_temp,
                      S10 = S10_temp)
    
    
    iter = iter + 1
    
    param_history_matrix[iter + 1,1] <- alpha_temp
    param_history_matrix[iter + 1,2] <- theta_temp
    param_history_matrix[iter + 1,3] <- g_temp
    param_history_matrix[iter + 1,4] <- sigma2_temp
    
    # debug
    print(param_history_matrix[iter + 1,])
    
  }
  
  
  returned_list <- list(alpha = alpha_temp, 
                        beta = beta_temp,
                        theta = theta_temp, 
                        g = g_temp,
                        sigma2 = sigma2_temp,
                        iter_history = param_history_matrix)
  
  if(is_fixed_effect){
    returned_list[["beta_iter_history"]] <- beta_iter_history
  }
  
  # return estimates
  return(returned_list)
}
