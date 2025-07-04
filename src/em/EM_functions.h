#include <RcppArmadillo.h>

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

arma::mat Omega_one_t(const arma::vec & vY_fixed_res_t,
                      const arma::vec & vZt,
                      const arma::mat & mXz,
                      const arma::mat & mPsmt,
                      double alpha);

arma::mat OmegaSumUpdate(const arma::mat & mY_fixed_res,
                         const arma::mat & Zt,
                         const arma::mat & mXz,
                         const arma::cube & cPsmt,
                         double alpha);

double brent_minimize(std::function<double(double)> f,
                      double ax, double bx, double cx,
                      double tol = 1e-8, int max_iter = 100);

double ThetaUpdate(const arma::mat& dist_matrix,
                   double g,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   double theta0,
                   int N,
                   double lower = 0.00001,
                   double upper = 10.0);

double Sigma2Update(const arma::mat& Omega_sum,
                    const int n,
                    const int T);

arma::vec BetaUpdate(const arma::cube& Xbeta,
                     const arma::mat& y,
                     const arma::mat& z,
                     double alpha,
                     const arma::mat& Xz,
                     const arma::mat& inv_mXbeta_sum);
