

arma::mat ExpCor(const arma::mat& mdist,
                 double theta);

double AlphaUpdate(const arma::mat & mY_fixed_res,
                   const arma::mat & mZ,
                   const arma::mat & mXz,
                   const arma::cube & cPsm);

arma::mat ComputeS00(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::vec & z0,
                     const arma::mat & P0);

arma::mat ComputeS11(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::mat & S00,
                     const arma::vec & z0,
                     const arma::mat & P0);

arma::mat ComputeS10(const arma::mat & smoothed_states,
                     const arma::cube & lagone_smoothed_covars,
                     const arma::vec & z0);

double gUpdate(const arma::mat & S00,
               const arma::mat & S10);

arma::mat Omega_one_t(const arma::mat & mY_fixed_res,
                      const arma::vec & vZt,
                      const arma::mat & mXz,
                      const arma::mat & mPsmt,
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

arma::vec BetaUpdate(const arma::cube& Xbeta,
                     const arma::mat& y,
                     const arma::mat& z,
                     double alpha,
                     const arma::mat& Xz,
                     const arma::mat& inv_mXbeta_sum);
