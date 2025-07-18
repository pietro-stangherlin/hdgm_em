rm(list = ls())

# temp script to try to understand where c++ corresponding code fails

# covariance specification ----------------------------
# where d is a distance matrix
# theta > 0
ExpCor <- function(mdist, theta){
  exp(-mdist/theta)
}

N <- 1000
Y_LEN <- 4

dist_matrix <- matrix(scan(text = "0   1.0255   0.4299   0.5974
   1.0255        0   0.6138   0.8521
   0.4299   0.6138        0   0.5979
   0.5974   0.8521   0.5979        0",
                           what = numeric(), quiet = T), ncol = Y_LEN)

S00 <- matrix(scan(text = "0.5524   0.0354   0.1927   0.1697
   0.0354   1.2121   0.2719  -0.4025
   0.1927   0.2719   0.5100   0.0273
   0.1697  -0.4025   0.0273   1.0155",
                   what = numeric(), quiet = T), ncol = Y_LEN)

S10 <- matrix(scan(text = "-0.0153   0.5307   0.0977  -0.4607
  -0.0503   1.6796   0.3097  -1.4570
  -0.0270   0.8917   0.1653  -0.7740
   0.0021  -0.0769  -0.0143   0.0672",
                   what = numeric(), quiet = T), ncol = Y_LEN)

S11 <- matrix(scan(text = "0.4770   1.5054   0.7996  -0.0688
   1.5054   4.7631   2.5294  -0.2176
   0.7996   2.5294   1.3447  -0.1156
  -0.0688  -0.2176  -0.1156   0.0111",
                   what = numeric(), quiet = T), ncol = Y_LEN)

g <- 0.6

theta0 <- 5
v0 <- 4


NegToOptimThetaV <- function(x){
  theta <- x[1]
  v <- x[2]

  Sigma_eta <- v * ExpCor(mdist = dist_matrix,
                      theta = theta)

  I = diag(nrow = NROW(S00))
  G = g * I

  return(
    N * determinant(x = Sigma_eta, logarithm = TRUE)$modulus +
      sum(diag(solve(Sigma_eta) %*%
                 (S11 - S10 %*% t(G) - G %*% t(S10)  + G %*% S00 %*% t(G)))))
}

eps <- 3
n_grid <- 100
theta_grid <- seq(1e-05, theta0 + eps, length = n_grid)
v_grid <- seq(1e-05, v0 + eps, length = n_grid)

z = matrix(NA, ncol = n_grid, nrow = n_grid)

for(i in 1:length(theta_grid)){
  for(j in 1:length(v_grid)){
  z[i,j] <- NegToOptimThetaV(c(theta_grid[i], v_grid[j]))
  }
}

contour(theta_grid, v_grid, z)
min(z)
index_min = which(z == min(z), arr.ind = T)
theta_grid[index_min[1,1]]
v_grid[index_min[1,2]]





