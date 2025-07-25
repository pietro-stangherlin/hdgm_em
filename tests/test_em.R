rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

source("tests/test_helper.R")

# Simulation 1 -----------------------------------
set.seed(123)

N <- 10000
Y_LEN <- 5
THETA <- 2
G <- 0.8
A <- 1
SIGMAY <- 0.1
SIGMAZ <- 0.1

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

y.matr <- LinGauStateSpaceSim(n_times = N,
                            obs_dim = Y_LEN,
                            state_dim = Y_LEN,
                            transMatr = G * diag(nrow = Y_LEN),
                            obsMatr = A * diag(nrow = Y_LEN),
                            stateCovMatr = ETA_MATRIX,
                            obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                            zeroState = rep(0, Y_LEN))$observations


# Testing ---------------------------------

# Starting from true values -------------------------

# structured EM
res_EM <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 v0 = SIGMAZ^2,
                 g0 = G,
                 sigma20 = SIGMAY^2,
                 Xbeta_in = NULL,
                 z0_in = NULL,
                 P0_in = NULL,
                 max_iter = 50, # increment
                 verbose = TRUE)

cbind(res_EM$par_history[,1], res_EM$par_history[,NCOL(res_EM$par_history)])

# unstructured EM
res_un_EM = UnstructuredEM(y = y.matr,
                           Phi_0 = G * diag(nrow = Y_LEN),
                           A_0 = A * diag(nrow = Y_LEN),
                           Q_0 = ETA_MATRIX,
                           R_0 = SIGMAY^2 * diag(nrow = Y_LEN),
                           x0_in = rep(0, Y_LEN),
                           P0_in = 5 * diag(nrow = Y_LEN),
                           max_iter = 200,
                           bool_mat = TRUE)

res_un_EM$Phi
res_un_EM$A
res_un_EM$Q
res_un_EM$R

# Starting from not true values -------------------------

res_EM_dist <- EMHDGM(y = y.matr,
                 dist_matrix = DIST_MATRIX,
                 alpha0 = A ,
                 beta0 = rep(0, 2),
                 theta0 = THETA,
                 v0 = SIGMAZ^2,
                 g0 = G, # assuming stationarity: this has to be in (-1,1)
                 sigma20 = SIGMAY^2 + 1,
                 Xbeta_in = NULL,
                 z0_in = NULL,
                 P0_in = NULL,
                 max_iter = 300, # increment
                 verbose = TRUE)

cbind(res_EM$par_history[,1], res_EM_dist$par_history[,NCOL(res_EM_dist$par_history)])

plot(res_EM_dist$par_history[1,])
plot(res_EM_dist$par_history[3,])

# unstrucured

res_un_EM_dist = UnstructuredEM(y = y.matr,
                           Phi_0 = 2 * G * diag(nrow = Y_LEN),
                           A_0 = A * diag(nrow = Y_LEN),
                           Q_0 = ETA_MATRIX,
                           R_0 = SIGMAY^2 * diag(nrow = Y_LEN),
                           x0_in = rep(0, Y_LEN),
                           P0_in = diag(nrow = Y_LEN),
                           max_iter = 5,
                           bool_mat = TRUE)

res_un_EM_dist$Phi
res_un_EM_dist$A
res_un_EM_dist$Q
res_un_EM_dist$R






