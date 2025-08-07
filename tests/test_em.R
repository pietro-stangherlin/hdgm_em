rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

source("../R/model_simulation_helper.R")

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

# Rcpp::sourceCpp("src/utils/data_handling.cpp",
#                 rebuild = TRUE)


# Data generation -----------------------------------
set.seed(123)

N <- 10000
Y_LEN <- 5
THETA <- 2
G <- 0.8
A <- 1
SIGMAY <- 1
SIGMAZ <- 1

# generate x coordinate
# generate y coordinate

POINTS <- matrix(NA,
                 nrow = Y_LEN,
                 ncol = 2)

POINTS[,1] <- runif(Y_LEN)
POINTS[,2] <- runif(Y_LEN)

DIST_MATRIX <- as.matrix(dist(POINTS))
# add noise
for (i in NCOL(DIST_MATRIX)){

}

diag(DIST_MATRIX) = 0

COR_MATRIX <- ExpCor(mdist = DIST_MATRIX,
                     theta = THETA)

ETA_MATRIX <- SIGMAZ^2 * COR_MATRIX # state covariance

StateSpaceRes <- LinGauStateSpaceSim(n_times = N,
                                     transMatr = G * diag(nrow = Y_LEN),
                                     obsMatr = A * diag(nrow = Y_LEN),
                                     stateCovMatr = ETA_MATRIX,
                                     obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                                     zeroState = rep(0, Y_LEN))

y.matr <- StateSpaceRes$observations
x.matr <- StateSpaceRes$states

# Covariates generation ------------------------------------------------------------

TRUE_FIXED_BETA <- c(2, -1, 4)

y.matr.with.fixed <- matrix(data = NA,
                            nrow = Y_LEN, ncol = N)

FIXED_EFFECTS_DESIGN_MATRIX <- array(data = NA,
                                     dim = c(Y_LEN,
                                             length(TRUE_FIXED_BETA),
                                             N))

for(i in 1:N){
  FIXED_EFFECTS_DESIGN_MATRIX[,,i] <- matrix(rnorm(Y_LEN * length(TRUE_FIXED_BETA)),
                                             nrow = Y_LEN, ncol = length(TRUE_FIXED_BETA))

  y.matr.with.fixed[,i] <- y.matr[,i] + as.vector(FIXED_EFFECTS_DESIGN_MATRIX[,,i] %*%
                                                    as.matrix(TRUE_FIXED_BETA))

}

# Testing ---------------------------------

# STRUCTURED -------------------------------------------------

# NOT MISSING Y --------------------------------------------

# Starting from true values -------------------------

# structured EM --------------------------------
res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                 x0_in = rep(0, Y_LEN),
                 P0_in = diag(1, nrow = Y_LEN),
                 max_iter = 50, # increment
                 verbose = TRUE,
                 bool_mat = FALSE,
                 is_fixed_effects = FALSE)

cbind(res_EM$par_history[,1], res_EM$par_history[,res_EM$niter])

plot(res_EM$par_history[1,], type = "l")
plot(res_EM$par_history[2,], type = "l")
plot(res_EM$par_history[3,], type = "l")
plot(res_EM$par_history[4,], type = "l")
plot(res_EM$par_history[5,], type = "l")

# Starting from not true values -------------------------
# structured -------------------------------------
res_EM_dist <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = 5 * A ,
                 beta0 = rep(0, 2),
                 theta0 = 5 * THETA,
                 g0 = 6 * G, # assuming stationarity: this has to be in (-1,1)
                 sigma20 = 2 * SIGMAY^2,
                 Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                 x0_in = rep(0, Y_LEN),
                 P0_in = 5 * diag(nrow = Y_LEN),
                 max_iter = 500, # increment
                 verbose = TRUE,
                 bool_mat = TRUE,
                 is_fixed_effects = FALSE)

# false starting values
cbind(res_EM$par_history[,1], res_EM_dist$par_history[,1], res_EM_dist$par_history[,res_EM_dist$niter])


par(mfrow = c(2,2))
plot(res_EM_dist$par_history[1,1:res_EM_dist$niter], type = "l",
     xlab = "iter",
     ylab = "A")
abline(h = A, col = "red")
plot(res_EM_dist$par_history[2,1:res_EM_dist$niter], type = "l",
     xlab = "iter",
     ylab = "Theta")
abline(h = THETA, col = "red")
plot(res_EM_dist$par_history[3,1:res_EM_dist$niter], type = "l",
     xlab = "iter",
     ylab = "G")
abline(h = G, col = "red")
plot(res_EM_dist$par_history[4,1:res_EM_dist$niter], type = "l",
     xlab = "iter",
     ylab = "SIGMAY^2")
abline(h = SIGMAY^2, col = "red")
par(mfrow = c(1,1))


# Covariates ---------------------------
# starting from true values ----------------------

res_EM_dep <- EMHDGM(y = y.matr.with.fixed,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = TRUE_FIXED_BETA,
                 theta0 = THETA,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in =FIXED_EFFECTS_DESIGN_MATRIX,
                 x0_in = rep(0, Y_LEN),
                 P0_in = diag(nrow = Y_LEN),
                 max_iter = 50, # increment
                 verbose = TRUE,
                 bool_mat = FALSE,
                 is_fixed_effects = TRUE)

res_EM_dep$beta_history
res_EM_dep$llik

# starting from NOT true values ----------------------

res_EM_dep_false <- EMHDGM(y = y.matr.with.fixed,
                     dist_matrix = DIST_MATRIX,
                     alpha0 = 3 * A,
                     g0 = G,
                     beta0 = 5 * TRUE_FIXED_BETA,
                     theta0 = 4 * THETA,
                     sigma20 = 4 * SIGMAY^2,
                     Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                     x0_in = rep(0, Y_LEN),
                     P0_in = 5 * diag(nrow = Y_LEN),
                     max_iter = 100, # increment
                     verbose = TRUE,
                     bool_mat = FALSE,
                     is_fixed_effects = TRUE)

res_EM_dep_false$beta_history[,res_EM_dep_false$niter]
cbind(res_EM_dep$par_history[,1],
      res_EM_dep_false$par_history[,res_EM_dep_false$niter])

par(mfrow = c(2,2))
plot(res_EM_dep_false$par_history[1,1:res_EM_dep_false$niter], type = "l")
plot(res_EM_dep_false$par_history[2,1:res_EM_dep_false$niter], type = "l")
plot(res_EM_dep_false$par_history[3,1:res_EM_dep_false$niter], type = "l")
plot(res_EM_dep_false$par_history[4,1:res_EM_dep_false$niter], type = "l")
par(mfrow = c(1,1))

res_EM_dep_false$llik


# MISSING Y --------------------------------------------
y.miss = y.matr


for(i in 1:N){
  for(j in 1:Y_LEN){
      if(runif(1) < 0.05){
        y.miss[j, i] = NaN
      }
  }
}

# starting from true values -------------------------------------------

res_EM_miss <- EMHDGM(y = y.miss,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                 x0_in = rep(0, Y_LEN),
                 P0_in = diag(1, nrow = Y_LEN),
                 max_iter = 50, # increment
                 verbose = TRUE,
                 bool_mat = FALSE,
                 is_fixed_effects = FALSE)

cbind(res_EM_miss$par_history[,1], res_EM_miss$par_history[,res_EM_miss$niter])

plot(res_EM_miss$par_history[1,], type = "l")
plot(res_EM_miss$par_history[2,], type = "l")
plot(res_EM_miss$par_history[3,], type = "l")
plot(res_EM_miss$par_history[4,], type = "l")
plot(res_EM_miss$par_history[5,], type = "l")

# starting from not true values ---------------------------------------------
res_EM_miss_dist <- EMHDGM(y = y.miss,
                      dist_matrix = DIST_MATRIX,
                      alpha0 = 5 * A ,
                      beta0 = rep(0, 2),
                      theta0 = 5 * THETA,
                      g0 = 6 * G, # assuming stationarity: this has to be in (-1,1)
                      sigma20 = 2 * SIGMAY^2,
                      Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                      x0_in = rep(0, Y_LEN),
                      P0_in = 5 * diag(nrow = Y_LEN),
                      max_iter = 500, # increment
                      verbose = TRUE,
                      bool_mat = TRUE,
                      is_fixed_effects = FALSE)

# false starting values
cbind(res_EM_miss$par_history[,1], res_EM_miss_dist$par_history[,1], res_EM_miss_dist$par_history[,res_EM_miss_dist$niter])


par(mfrow = c(2,2))
plot(res_EM_miss_dist$par_history[1,1:res_EM_miss_dist$niter], type = "l",
     xlab = "iter",
     ylab = "A")
abline(h = A, col = "red")
plot(res_EM_miss_dist$par_history[2,1:res_EM_miss_dist$niter], type = "l",
     xlab = "iter",
     ylab = "Theta")
abline(h = THETA, col = "red")
plot(res_EM_miss_dist$par_history[3,1:res_EM_miss_dist$niter], type = "l",
     xlab = "iter",
     ylab = "G")
abline(h = G, col = "red")
plot(res_EM_miss_dist$par_history[4,1:res_EM_miss_dist$niter], type = "l",
     xlab = "iter",
     ylab = "SIGMAY^2")
abline(h = SIGMAY^2, col = "red")
par(mfrow = c(1,1))

# Covariates ---------------------------

y.miss.with.fixed = y.matr.with.fixed

for(i in 1:N){
  for(j in 1:Y_LEN){
    if(runif(1) < 0.05){
      y.miss.with.fixed[j, i] = NaN
    }
  }
}

# starting from true values ----------------------

res_EM_miss_dep <- EMHDGM(y = y.miss.with.fixed,
                     dist_matrix = DIST_MATRIX,
                     alpha0 = A,
                     beta0 = TRUE_FIXED_BETA,
                     theta0 = THETA,
                     g0 = G,
                     sigma20 = SIGMAY^2,
                     Xbeta_in =FIXED_EFFECTS_DESIGN_MATRIX,
                     x0_in = rep(0, Y_LEN),
                     P0_in = diag(nrow = Y_LEN),
                     max_iter = 50, # increment
                     verbose = TRUE,
                     bool_mat = FALSE,
                     is_fixed_effects = TRUE)

res_EM_miss_dep$beta_history
res_EM_miss_dep$llik

# starting from NOT true values ----------------------

res_EM_miss_dep_false <- EMHDGM(y = y.miss.with.fixed,
                           dist_matrix = DIST_MATRIX,
                           alpha0 = 3 * A,
                           g0 = G,
                           beta0 = 5 * TRUE_FIXED_BETA,
                           theta0 = 4 * THETA,
                           sigma20 = 4 * SIGMAY^2,
                           Xbeta_in = FIXED_EFFECTS_DESIGN_MATRIX,
                           x0_in = rep(0, Y_LEN),
                           P0_in = 5 * diag(nrow = Y_LEN),
                           max_iter = 100, # increment
                           verbose = TRUE,
                           bool_mat = FALSE,
                           is_fixed_effects = TRUE)

res_EM_miss_dep_false$beta_history[,res_EM_miss_dep_false$niter]
cbind(res_EM_miss_dep$par_history[,1],
      res_EM_miss_dep_false$par_history[,res_EM_miss_dep_false$niter])

par(mfrow = c(2,2))
plot(res_EM_miss_dep_false$par_history[1,1:res_EM_miss_dep_false$niter], type = "l")
plot(res_EM_miss_dep_false$par_history[2,1:res_EM_miss_dep_false$niter], type = "l")
plot(res_EM_miss_dep_false$par_history[3,1:res_EM_miss_dep_false$niter], type = "l")
plot(res_EM_miss_dep_false$par_history[4,1:res_EM_miss_dep_false$niter], type = "l")
par(mfrow = c(1,1))

res_EM_miss_dep_false$llik



