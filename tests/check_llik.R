rm(list = ls())
library(Rcpp)
library(RcppArmadillo)

source("tests/test_helper.R")

Rcpp::sourceCpp("src/kalman/Kalman_wrapper.cpp")


# generate some data -------------------
N <- 1000
Y_LEN <- 10
THETA <- 5
G <- 0.8
A <- 3
SIGMAY <- 0.1
SIGMAZ <- 2

# generate x coordinate
# generate y coordinate

POINTS <- matrix(NA,
                 nrow = Y_LEN,
                 ncol = 2)

POINTS[,1] <- runif(Y_LEN)
POINTS[,2] <- runif(Y_LEN)

DIST_MATRIX <- as.matrix(dist(POINTS))

diag(DIST_MATRIX) = 0

COR_MATRIX <- ExpCor(mdist = DIST_MATRIX,
                     theta = THETA)

ETA_MATRIX <- SIGMAZ^2 * COR_MATRIX # state covariance

y.matr <- LinGauStateSpaceSim(n_times = N,
                              obs_dim = Y_LEN,
                              state_dim = Y_LEN,
                              transMatr = G * diag(nrow = Y_LEN),
                              obsMatr = A * diag(nrow = Y_LEN),
                              stateCovMatr = ETA_MATRIX,
                              obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                              zeroState = rep(0, Y_LEN))$observations

# Theta and V fixed the others ----------------------------------------

N_grid = 30

alpha_vals = seq(A - 0.5, A + 0.5, length = N_grid)
sigma2z_vals = seq(SIGMAZ^2 - 1, SIGMAZ^2 + 1, length = N_grid)

llik_vals <- matrix(NA, N_grid, N_grid)


counter = 0
for(i in 1:N_grid){
  for(j in 1:N_grid){

    if((counter %% 100) == 0){
      print(counter)
    }

    llik_vals[i,j] <- SKF(Y = y.matr,
                          Phi = G * diag(nrow = Y_LEN),
                          A = alpha_vals[i] * diag(nrow = Y_LEN),
                          Q = sigma2z_vals[j] * COR_MATRIX,
                          R = SIGMAY^2 * diag(nrow = Y_LEN),
                          x_0 = rep(0, Y_LEN),
                          P_0 = diag(1, Y_LEN),
                          retLL = TRUE)$loglik
    counter = counter + 1

  }
}

contour(alpha_vals, sigma2z_vals, llik_vals,
        xlab = "alpha",
        ylab = "sigma2z",
        levels = quantile(llik_vals, c(0.7, 0.8, 0.9, 0.95, 0.975, 0.99)),
        main = "llik")

max_index <- which(llik_vals == max(llik_vals),
                   arr.ind = T)

points(alpha_vals[max_index[1]],
       sigma2z_vals[max_index[2]],
       pch = 16, col = "red")

points(A, SIGMAZ^2,
       pch = 16, col = "blue")


# Optim llik via optim + Kalman ---------------------------------------------

FunctionToOptim <- function(param){
 log_theta = param[1]
 g = param[2]
 a = param[3]
 log_sigma2y = param[4]
 log_sigma2z = param[5]

 return(-SKF(Y = y.matr,
             Phi = g * diag(nrow = Y_LEN),
             A = a * diag(nrow = Y_LEN),
             Q = exp(log_sigma2y) * ExpCor(mdist = DIST_MATRIX,
                                           theta = exp(log_theta)),
             R = exp(log_sigma2z) * diag(nrow = Y_LEN),
             x_0 = rep(0, Y_LEN),
             P_0 = diag(1, Y_LEN),
             retLL = TRUE)$loglik)
}


res_optim <-optim(par = c(log(THETA), G, A,
                          log(SIGMAY^2), log(SIGMAZ^2)),
                  fn = FunctionToOptim)

res_true_scale = c(exp(res_optim$par[1]), res_optim$par[2], res_optim$par[3],
  exp(res_optim$par[4]), exp(res_optim$par[5]))

cbind(c(THETA, G, A, SIGMAY^2, SIGMAZ^2), res_true_scale)



