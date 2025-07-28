rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

Sys.setenv("PKG_CXXFLAGS"="-std=c++20")

Rcpp::sourceCpp("src/em/EM_wrapper.cpp",
                rebuild = TRUE)

# Rcpp::sourceCpp("src/utils/data_handling.cpp",
#                 rebuild = TRUE)

source("tests/test_helper.R")

# Simulation function unstructured ----------------------------------------

SimulEMUn <- function(B,
                    n_times,
                    transMatr, # true model parameters
                    obsMatr,
                    stateCovMatr,
                    obsCovMatr,
                    zeroState,
                    em_transMatr, # starting EM values
                    em_obsMatr,
                    em_stateCovMatr,
                    em_obsCovMatr,
                    em_zeroState,
                    em_zeroStateCov,
                    em_max_iter = 50,
                    em_bool_mat = TRUE,
                    em_verbose = FALSE){

  p = NROW(transMatr)
  q = NROW(obsMatr)

  Phi_hat_array <- array(NA, dim = c(p, p, B))
  A_hat_array <- array(NA, dim = c(q, p, B))
  Q_hat_array <- array(NA, dim = c(p, p, B))
  R_hat_array <- array(NA, dim = c(q, q, B))
  P0_smooth_hat_array <- array(NA, dim = c(p, p, B))
  x0_hat_smooth <- matrix(NA, p, B)
  llik_hat_vec <- rep(NA, B)


  for(b in 1:B){

    if(b %% 50 == 0){
      print(paste0("simulation iter: ", b, collapse = ""))
    }

    y.matr <- LinGauStateSpaceSim(n_times = n_times,
                                  transMatr = transMatr,
                                  obsMatr = obsMatr,
                                  stateCovMatr = stateCovMatr,
                                  obsCovMatr = obsCovMatr,
                                  zeroState = zeroState)$observations

    res_un_EM <- UnstructuredEM(y = y.matr,
                                Phi_0 = em_transMatr,
                                A_0 = em_obsMatr,
                                Q_0 = em_stateCovMatr,
                                R_0 = em_obsCovMatr,
                                x0_in = em_zeroState,
                                P0_in = em_zeroStateCov,
                                max_iter = em_max_iter,
                                bool_mat = em_bool_mat,
                                verbose = em_verbose)

    Phi_hat_array[,,b] <- res_un_EM[["Phi"]]
    A_hat_array[,,b] <- res_un_EM[["A"]]
    Q_hat_array[,,b] <- res_un_EM[["Q"]]
    R_hat_array[,,b] <- res_un_EM[["R"]]
    P0_smooth_hat_array[,,b] <- res_un_EM[["P0_smoothed"]]
    x0_hat_smooth[,b] <- res_un_EM[["x0_smoothed"]]

  }

  return(list("Phi" = Phi_hat_array,
              "A" = A_hat_array,
              "Q" = Q_hat_array,
              "R" = R_hat_array,
              "x0_smoothed" = x0_hat_smooth,
              "P0_smoothed" = P0_smooth_hat_array))

}


# Simulation 1 -----------------------------------
set.seed(123)

N <- 10000
Y_LEN <- 10
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

# Covariates ------------------------------------------------------------

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

# unstructured EM -------------------------------------
res_un_EM = UnstructuredEM(y = y.matr,
                           Phi_0 = G * diag(nrow = Y_LEN),
                           A_0 = A * diag(nrow = Y_LEN),
                           Q_0 = ETA_MATRIX,
                           R_0 = SIGMAY^2 * diag(nrow = Y_LEN),
                           x0_in = rep(0, Y_LEN),
                           P0_in = ETA_MATRIX,
                           max_iter = 200,
                           bool_mat = FALSE,
                           verbose = TRUE)

res_un_EM$Phi
res_un_EM$A
res_un_EM$Q
res_un_EM$R
res_un_EM$x0_smoothed
res_un_EM$P0_smoothed


# simulation -------------------------------------------
res_em_sim_true_start <- SimulEMUn(B = 300,
                                   n_times = N,
                                   transMatr = G * diag(nrow = Y_LEN),
                                   obsMatr = A * diag(nrow = Y_LEN),
                                   stateCovMatr = ETA_MATRIX,
                                   obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                                   zeroState = rep(0, Y_LEN),
                                   em_transMatr = G * diag(nrow = Y_LEN),
                                   em_obsMatr = A * diag(nrow = Y_LEN),
                                   em_stateCovMatr = ETA_MATRIX,
                                   em_obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                                   em_zeroState = rep(0, Y_LEN),
                                   em_zeroStateCov = ETA_MATRIX,
                                   em_max_iter = 50,
                                   em_bool_mat = TRUE,
                                   em_verbose = FALSE)


b = dim(res_em_sim_true_start$Phi)[3]

plot(1:b, rep(A, b), type = "l")
lines(res_em_sim_true_start$A[,,], col = "red")

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
                 max_iter = 200, # increment
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

# unstrucured ----------------------------------------------

res_un_EM_dist = UnstructuredEM(y = y.matr,
                           Phi_0 = 2 * G * diag(nrow = Y_LEN),
                           A_0 = A * diag(nrow = Y_LEN),
                           Q_0 = ETA_MATRIX,
                           R_0 = SIGMAY^2 * diag(nrow = Y_LEN),
                           x0_in = rep(0, Y_LEN),
                           P0_in = 5 * diag(nrow = Y_LEN),
                           max_iter = 500,
                           bool_mat = FALSE,
                           verbose = TRUE)

res_un_EM_dist$Phi
res_un_EM_dist$A
res_un_EM_dist$Q
res_un_EM_dist$R

res_un_EM_dist$x0_smoothed


res_em_sim_false_start <- SimulEMUn(B = 50,
                                   n_times = N,
                                   transMatr = G * diag(nrow = Y_LEN),
                                   obsMatr = A * diag(nrow = Y_LEN),
                                   stateCovMatr = ETA_MATRIX,
                                   obsCovMatr = SIGMAY^2 * diag(nrow = Y_LEN),
                                   zeroState = rep(0, Y_LEN),
                                   em_transMatr = 2 * G * diag(nrow = Y_LEN),
                                   em_obsMatr = A * diag(nrow = Y_LEN),
                                   em_stateCovMatr = 2 * ETA_MATRIX,
                                   em_obsCovMatr = 2 * SIGMAY^2 * diag(nrow = Y_LEN),
                                   em_zeroState = 3 * rep(0, Y_LEN),
                                   em_zeroStateCov = 3 *  ETA_MATRIX,
                                   em_max_iter = 50,
                                   em_bool_mat = FALSE,
                                   em_verbose = FALSE)

b = dim(res_em_sim_false_start$Phi)[3]

plot(1:b, rep(A, b), type = "l", ylim = c(A - 4, A + 4))
lines(res_em_sim_false_start$A[1,1,], col = "red")
lines(res_em_sim_false_start$A[2,2,], col = "red")
lines(res_em_sim_false_start$A[3,3,], col = "red")


plot(1:b, rep(G, b), type = "l", ylim = c(G - 4, G + 4))
lines(res_em_sim_false_start$Phi[1,1,], col = "red")
lines(res_em_sim_false_start$Phi[2,2,], col = "red")
lines(res_em_sim_false_start$Phi[3,3,], col = "red")

plot(1:b, rep(0, b), type = "l", ylim = c(G - 4, G + 4))
lines(res_em_sim_false_start$Phi[1,2,], col = "red")
lines(res_em_sim_false_start$Phi[2,3,], col = "red")
lines(res_em_sim_false_start$Phi[3,5,], col = "red")

plot(1:b, rep(ETA_MATRIX[1,1], b), type = "l", ylim = c(G - 4, G + 4))
lines(res_em_sim_false_start$Q[1,1,], col = "red")

plot(1:b, rep((SIGMAY^2 * diag(nrow = Y_LEN))[1,1], b), type = "l", ylim = c(G - 4, G + 4))
lines(res_em_sim_false_start$R[1,1,], col = "red")

plot(1:b, res_em_sim_false_start$x0_smoothed[1,], type = "l")
apply(res_em_sim_false_start$x0_smoothed, 1, mean)
apply(res_em_sim_false_start$x0_smoothed, 1, sd)

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


