

arma::mat ExpCor(const arma::mat& mdist, double theta);

double AlphaUpdate(arma::mat & mY_fixed_res,
                  arma::mat & mZ,
                  arma::mat & mXz,
                  arma::cube & cPsm);

arma::mat ComputeS00(arma::mat & smoothed_states,
                     arma::cube & smoothed_vars,
                     arma::vec & z0,
                     arma::mat & P0);

arma::mat ComputeS11(arma::mat & smoothed_states,
                     arma::cube & smoothed_vars,
                     arma::mat & S00,
                     arma::vec & z0,
                     arma::mat & P0);

arma::mat ComputeS10(arma::mat & smoothed_states,
                     arma::cube & lagone_smoothed_covars,
                     arma::vec & z0);

double gUpdate(arma::mat & S00,
               arma::mat & S10);

arma::mat Omega_one_t(arma::mat & mY_fixed_res,
                      arma::vec & vZt,
                      arma::mat & mXz,
                      arma::mat & mPsmt,
                      double alpha);

double brent_optimize(const std::function<double(double)>& f,
                      double lower,
                      double upper,
                      double tol = 1e-5,
                      int max_iter = 100);

double ThetaUpdate(const arma::mat& dist_matrix,
                   double g,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   double theta0,
                   int N,
                   double lower = 0.00001,
                   double upper = 10.0);

arma::vec BetaUpdate(const std::vector<arma::mat>& Xbeta,
                     const arma::mat& y,
                     const arma::mat& z,
                     double alpha,
                     const arma::mat& Xz,
                     const arma::mat& inv_mXbeta_sum);
