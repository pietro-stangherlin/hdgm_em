rm(list = ls())

source("R/model_simulation_helper.R")
source("R/bootstrap_helper.R")


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


# Bootstrap ----------------------------------------------
# here assuming mles are in true values

boot.res <- BootstrapHDGM(mle.structural = c(A, G, THETA, SIGMAY^2),
                          mle.beta.fixed = TRUE_FIXED_BETA,
                          y.matr = y.matr,
                          dist.matr = DIST_MATRIX,
                          X.array = FIXED_EFFECTS_DESIGN_MATRIX,
                          zero_state = rep(0, NROW(y.matr)),
                          zero_state_var = diag(1, nrow = NROW(y.matr)),
                          max_EM_iter = 30,
                          start_obs_index = 1,
                          B = 10)

boot.res

