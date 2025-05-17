rm(list = ls())

source("HDGM_sim_script.R")

require(MARSS)

# simulate HDGM
# and then estimate its state variables using KFAS

# Simulation 1 -----------------------------------
set.seed(123)

N <- 1000
Y_LEN <- 2
THETA <- 5
G <- 0.8
UPSILON <- 3
SIGMA <- 0.1

DIST_MATRIX <- matrix(c(0, 1,
                        1, 0), 
                      ncol = Y_LEN,
                      nrow = Y_LEN)

COR_MATRIX <- ExpCor(mdist = DIST_MATRIX,
                     theta = THETA)

res <- RHDGM(n = N,
             y_len = Y_LEN,
             cor_matr = COR_MATRIX,
             sigmay = SIGMA,
             upsilon = UPSILON,
             gHDGM = G,
             z0 = rep(0, Y_LEN))

y.matr <- res$y


# Calculli Test -------------------------------------------------
source("calculli_2015_HDGM_EM.R")


# starting from true values
# should be stuck at maximum since EM is greedy
# (if MLE is consistent)
em_res <- EMHDGM(y = t(y.matr),
                 Xbeta = NULL,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = UPSILON,
                 beta0 = rep(0, 2),
                 g0 = G,
                 sigma20 = SIGMA^2,
                 theta0 = THETA,
                 z0 = as.matrix(rep(0, 2)),
                 max_iter = 10) # low iter number due to slowiness

em_res$iter_history

# starting != MLE


# perturbation term
epsilon <- 0.5

# slow convergence
em_res_other_start <- EMHDGM(y = t(y.matr),
                             Xbeta = NULL,
                             dist_matrix = DIST_MATRIX,
                             alpha0 = UPSILON + epsilon,
                             beta0 = rep(0, 2),
                             g0 = G + epsilon,
                             sigma20 = SIGMA^2 + epsilon,
                             theta0 = THETA + epsilon,
                             z0 = as.matrix(rep(0, 2)),
                             max_iter = 100,
                             verbose = FALSE)

em_res_other_start$iter_history


# now try simulating using covariates
# two covariates: i.e one intercept
beta_true_matrix <- as.matrix(c(1,1))

my.Xbeta <- array(NA, dim = c(2,2,N))
for(i in 1:N){
  my.Xbeta[,,i] <- diag(1,2)
}

fixed_intercepts_matr <- matrix(NA, nrow = N, ncol = 2)
for(i in 1:N){
  fixed_intercepts_matr[i,] <- my.Xbeta[,,i] %*% beta_true_matrix
}


y.matr.fixed <- y.matr + fixed_intercepts_matr


em_res_fixed <- EMHDGM(y = t(y.matr.fixed),
                       Xbeta = my.Xbeta,
                       dist_matrix = DIST_MATRIX,
                       alpha0 = UPSILON,
                       beta0 = beta_true_matrix,
                       g0 = G,
                       sigma20 = SIGMA^2,
                       theta0 = THETA,
                       z0 = as.matrix(rep(0, 2)),
                       max_iter = 10,
                       verbose = FALSE)

em_res_fixed$iter_history
em_res_fixed$beta
