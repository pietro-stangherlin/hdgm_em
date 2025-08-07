Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

# if already executed EM
load("data/HDGM_res_EM.RData")


# Initial linear model beta estimates -------------------------------------

load("data/agri_df.RData")

lm.formula <- paste0("AQ_pm25~",
                     paste0("Month+",
                            paste0(selected_vars_names, collapse = "+"),
                            collapse = ""),
                     collapse = "")

lm.fit <- lm(as.formula(lm.formula), data = agrim_df)
rm(agrim_df)

# EM ---------------------------------------------------
load("data/agri_matrix_array_em.RData")

res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = dists_matr,
                 alpha0 = 1,
                 beta0 = coef(lm.fit), # start with OLS estimate
                 theta0 = 1,
                 g0 = 0.5,
                 sigma20 = 1,
                 Xbeta_in = X.array,
                 x0_in = rep(0, q),
                 P0_in = diag(1, nrow = q),
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
          method.args = list(eps=1e-3, d=0.1, zero.tol=sqrt(.Machine$double.eps/7e-7), r=4, v=2, show.details=FALSE))

# asymptotic information matrix
asymptotic_var <- solve(-hess.hat) / NCOL(y.matr)

# get an idea of dispersion
cbind(c(res_EM$par_history[,res_EM$niter],
        res_EM$beta_history[,res_EM$niter]),
      sqrt(diag(asymptotic_var)))

# Save Results -------------------

save(res_EM, asymptotic_var, file = "data/HDGM_res_EM.RData")

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
                          start_obs_index = 10,
                          B = 10)








