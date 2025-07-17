rm(list = ls())

# temp script to try to understand where c++ corresponding code fails

# covariance specification ----------------------------
# where d is a distance matrix
# theta > 0
ExpCor <- function(mdist, theta){
  exp(-mdist/theta)
}

N <- 1000

dist_matrix <- matrix(c(0, 1,
                        1, 0), ncol = 2)

S00 <- matrix(c(0.5250,   0.2839,
                0.2839,   0.4948),
              ncol = 2)

S10 <- matrix(c(0.1168,  0.0651,
                0.0959,   0.0550),
              ncol = 2)

S11 <- matrix(c(0.3060,   0.2530,
                0.2530,   0.2110),
              ncol = 2)

g <- 0.8

theta0 <- 5
v0 <- 1


NegToOptimThetaV <- function(x){
  theta <- x[1]
  v <- x[2]

  Sigma_eta <- v * ExpCor(mdist = dist_matrix,
                      theta = theta)
  return(
    N * determinant(x = Sigma_eta, logarithm = TRUE)$modulus +
      sum(diag(solve(Sigma_eta) %*%
                 (S11 - g * S10 - g * t(S10)  + g^2 * S00))))
}

n_grid <- 100
theta_grid <- seq(theta0 - 2, theta0 + 2, length = n_grid)
v_grid <- seq(1e-5, v0 + 2, length = n_grid)

z = matrix(NA, ncol = n_grid, nrow = n_grid)

for(i in 1:length(theta_grid)){
  for(j in 1:length(v_grid)){
  z[i,j] <- NegToOptimThetaV(c(theta_grid[i], v_grid[j]))
  }
}

contour(theta_grid, v_grid, z)









