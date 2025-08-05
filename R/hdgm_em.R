Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)




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
source("llik_helper.R")

# bad alloc :(
# hess <- numDeriv::hessian(func = HDGM.Llik,
#                   x = c(res_EM$par_history[,res_EM$niter],
#                     res_EM$beta_history[,res_EM$niter]),
#                   y.matr = y.matr,
#           dist.matr = dists_matr_sub,
#           X.array = X.array)

# Save Results -------------------

save(res_EM, file = "data/HDGM_res_EM.RData")

