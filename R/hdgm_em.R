Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

# if already executed EM
load("data/HDGM_res_EM.RData")
load("data/HDGM_cv_res.RData")


# Initial linear model beta estimates -------------------------------------

load("data/agri_df.RData")
load("data/agri_matrix_array_em.RData")

lm.formula <- paste0("AQ_pm25~",
                     paste0("Month+",
                            paste0(selected_vars_names, collapse = "+"),
                            collapse = ""),
                     collapse = "")

lm.fit.agri <- lm(as.formula(lm.formula), data = agrim_df)

# lm.formula.week <- paste0("AQ_pm25~",
#                      paste0("Month+WeekDay",
#                             paste0(selected_vars_names, collapse = "+"),
#                             collapse = ""),
#                      collapse = "")
#
# lm.fit.agri.week <- lm(as.formula(lm.formula.week), data = agrim_df)

coef_names <- names(coef(lm.fit.agri))

rm(agrim_df)

# EM ---------------------------------------------------

res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = dists_matr,
                 alpha0 = 5,
                 beta0 = coef(lm.fit.agri), # else start with OLS estimate
                 theta0 = 1,
                 g0 = 0.5,
                 sigma20 = 5,
                 Xbeta_in = X.array,
                 x0_in = rep(0, nrow(y.matr)),
                 P0_in = diag(1, nrow = nrow(y.matr)),
                 rel_llik_tol = 1e-5,
                 max_iter = 200,
                 verbose = TRUE,
                 bool_mat = TRUE,
                 is_fixed_effects = TRUE)

res_EM_scale <- EMHDGM_statescale(y = y.matr,
                                  dist_matrix = dists_matr,
                                  alpha0 = 1,
                                  beta0 = coef(lm.fit.agri),
                                  theta0 = 1,
                                  g0 = 0.5,
                                  sigma20 = 5,
                                  sigma2state0 = 5,
                                  Xbeta_in = X.array,
                                  x0_in = rep(0, nrow(y.matr)),
                                  P0_in = diag(1, nrow = nrow(y.matr)),
                                  max_iter = 200, # increment
                                  verbose = TRUE,
                                  var_terminating_lim = 0.1,
                                  bool_mat = TRUE,
                                  is_fixed_effects = TRUE)

res_EM_diag <- EMHDGM_diag(y = y.matr,
                 dist_matrix = dists_matr,
                 alpha0 = 5,
                 beta0 = coef(lm.fit.agri), # else start with OLS estimate
                 theta0 = 1,
                 g0 = 0.5,
                 sigma20 = 5,
                 Xbeta_in = X.array,
                 x0_in = rep(0, nrow(y.matr)),
                 P0_in = diag(1, nrow = nrow(y.matr)),
                 rel_llik_tol = 1e-5,
                 max_iter = 200,
                 verbose = TRUE,
                 bool_mat = TRUE,
                 is_fixed_effects = TRUE)

# Asymptotic Std -----------------
# not supported at the moment
source("R/llik_helper.R")

# bad alloc :(
hess.hat <- numDeriv::hessian(func = HDGM.Llik,
                  x = c(res_EM$par_history[,res_EM$niter],
                    res_EM$beta_history[,res_EM$niter]),
                  y.matr = y.matr,
          dist.matr = dists_matr,
          X.array = X.array,
          method = "Richardson", # WARNING: decrease eps (but it's REALLY slow)
          method.args = list(eps=1e-2, d=0.1, zero.tol=sqrt(.Machine$double.eps/7e-7), r=4, v=2, show.details=FALSE))

# asymptotic information matrix
asymptotic_var <- solve(-hess.hat)

# get an idea of dispersion
round(cbind(c(res_EM$par_history[,res_EM$niter],
        res_EM$beta_history[,res_EM$niter]),
      sqrt(diag(asymptotic_var))),3)

# Save Results -------------------

save(res_EM, res_EM_scale, asymptotic_var, coef_names, file = "data/HDGM_res_EM.RData")

# Bootstrap ----------------------
source("R/bootstrap_helper.R")

# not working at the moment
boot.res <- BootstrapHDGM(mle.structural = res_EM$par_history[,res_EM$niter],
                          mle.beta.fixed = res_EM$beta_history[,res_EM$niter],
                          y.matr = y.matr,
                          dist.matr = dists_matr,
                          X.array = X.array,
                          zero_state = rep(0, NROW(y.matr)),
                          zero_state_var = diag(1, nrow = NROW(y.matr)),
                          max_EM_iter = 30,
                          start_obs_index = 1,
                          B = 10)

save(boot.res, file = "data/HDGM_boot_res.RData")

# CV ---------------------------------
source("R/cv_helper.R")

# Expanding Window ---------

cv_expw_res <- CVExpandingWindow(y_matr = y.matr,
                                 X_array = X.array,
                                 dist_matr = dists_matr,
                                 initial_est_structural = res_EM$par_history[,res_EM$niter],
                                 intial_est_fixed = res_EM$beta_history[,res_EM$niter],
                                 starting_obs = 1824,
                                 step_ahead_pred = 3,
                                 max_EM_iter = 50)

# LOSO ---------------------

cv_stations_id <- 1:49

cv_loso_res <- CVLOSO(y_matr = y.matr,
                      X_array = X.array,
                      dist_matr = dists_matr,
                      initial_est_structural = res_EM$par_history[,res_EM$niter],
                      intial_est_fixed = res_EM$beta_history[,res_EM$niter],
                      validation_station_index = cv_stations_id,
                      max_EM_iter = 50)

colnames(cv_loso_res) <- cv_stations_id

MSE <- function(x){
  mean( (x - mean(x, na.rm = TRUE))^2 , na.rm = TRUE)
}

MSE_zero_mean <- function(x){
  mean( x^2 , na.rm = TRUE)
}

sqrt(apply(cv_loso_res, 2, MSE_zero_mean))
sqrt(apply(cv_expw_res, 1, MSE_zero_mean))

plot(y.matr[1,], type = "l", ylim = c(-30, 100))
lines(cv_loso_res[,1], col = 2)


# Fixed effects residuals --------------------------------
# with parameters estimated on all data, not CV

fixed_effects_res <- matrix(NA, nrow = nrow(y.matr), ncol = ncol(y.matr))
for(i in 1:NCOL(y.matr)){
  fixed_effects_res[,i] <- y.matr[,i] - X.array[,,i] %*% res_EM$beta_history[,res_EM$niter]
}


plot(y.matr[1,], type = "l", ylim = c(-30, 100))
lines(fixed_effects_res[1,], col = 2)

sqrt(apply(fixed_effects_res, 1, MSE_zero_mean))

save(fixed_effects_res, cv_expw_res, cv_loso_res, file = "data/HDGM_cv_res.RData")


