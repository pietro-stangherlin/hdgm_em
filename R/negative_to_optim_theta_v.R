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

S00 <- matrix(scan(text = "9.7827e+03   7.3391e+03   8.6617e+03   8.6357e+03
   7.3391e+03   9.0619e+03   7.8139e+03   7.7856e+03
   8.6617e+03   7.8139e+03   9.1350e+03   8.3127e+03
   8.6357e+03   7.7856e+03   8.3127e+03   1.0061e+04",
                   what = numeric(), quiet = T), ncol = Y_LEN)

S10 <- matrix(scan(text = "7.5570e+03   5.4399e+03   6.6008e+03   6.6468e+03
   5.6991e+03   6.9880e+03   6.0478e+03   6.0523e+03
   6.6603e+03   5.7916e+03   6.9481e+03   6.3184e+03
   6.6221e+03   5.8304e+03   6.3416e+03   7.7916e+03",
                   what = numeric(), quiet = T), ncol = Y_LEN)

S11 <- matrix(scan(text = "9.7932e+03   7.3596e+03   8.6792e+03   8.6521e+03
   7.3596e+03   9.0991e+03   7.8467e+03   7.8169e+03
   8.6792e+03   7.8467e+03   9.1630e+03   8.3393e+03
   8.6521e+03   7.8169e+03   8.3393e+03   1.0085e+04",
                   what = numeric(), quiet = T), ncol = Y_LEN)

g <- 0.8

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

eps <- 0.3
n_grid <- 100
theta_grid <- seq(theta0 - eps, theta0 + eps, length = n_grid)
v_grid <- seq(v0 - eps, v0 + eps, length = n_grid)

z = matrix(NA, ncol = n_grid, nrow = n_grid)

for(i in 1:length(theta_grid)){
  for(j in 1:length(v_grid)){
  z[i,j] <- NegToOptimThetaV(c(theta_grid[i], v_grid[j]))
  }
}

contour(theta_grid, v_grid, z)
points(theta0, v0, col = "blue", pch = 16)
min(z)
index_min = which(z == min(z), arr.ind = T)
theta_grid[index_min[1,1]]
v_grid[index_min[1,2]]
points(theta_grid[index_min[1,1]], v_grid[index_min[1,2]],
       col = "red", pch = 16)




