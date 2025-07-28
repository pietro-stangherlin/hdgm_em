#pragma once

#include <RcppArmadillo.h>
#include "../utils/symmetric_matr_vec.h"

template <typename CovStore>
arma::mat ComputeS00_core(const arma::mat & smoothed_states,
                                     const CovStore & smoothed_vars,
                                     const arma::vec & z0_smoothed,
                                     const arma::mat & P0_smoothed);


template <typename CovStore>
arma::mat ComputeS11_core(const arma::mat & smoothed_states,
                          CovStore & smoothed_vars,
                          const arma::mat & S00,
                          const arma::vec & z0_smoothed,
                          const arma::mat & P0_smoothed);


template <typename CovStore>
arma::mat ComputeS10_core(const arma::mat & smoothed_states,
                          const CovStore & lagone_smoothed_covars,
                          const arma::vec & z0_smoothed);

template <typename CovStore>
double AlphaUpdate_core(const arma::mat & mY_fixed_res,
                        const arma::mat & mZ,
                        const arma::mat & mXz,
                        const CovStore & cPsm);

template <typename CovStore>
arma::mat OmegaSumUpdate_core(const arma::mat & mY_fixed_res,
                                         const arma::mat & Zt,
                                         const arma::mat & mXz,
                                         const CovStore & cPsmt,
                                         double alpha);




double gUpdate(const arma::mat & S00,
               const arma::mat & S10);

arma::mat Omega_one_t(const arma::vec & vY_fixed_res_t,
                      const arma::vec & vZt,
                      const arma::mat & mXz,
                      const arma::mat & mPsmt,
                      double alpha);

double LogThetaNegativeToOptim(const double log_theta,
                               const arma::mat &dist_matrix,
                               const arma::mat &S00,
                               const arma::mat &S10,
                               const arma::mat &S11,
                               const arma::mat &Phi,
                               const int &T);

double ThetaUpdate(const arma::mat &dist_matrix,
                   const arma::mat &Phi,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   int &T);

double theta_v_negative_to_optim_log_scale(const std::array<double,2>& log_theta_v,
                                           const arma::mat& dist_matrix,
                                           const arma::mat& S00,
                                           const arma::mat& S10,
                                           const arma::mat& S11,
                                           const double& g,
                                           const int& N);

std::array<double,2> ThetaVUpdate(const arma::mat& dist_matrix,
                                  double& g,
                                  int& N,
                                  const arma::mat& S00,
                                  const arma::mat& S10,
                                  const arma::mat& S11,
                                  const std::array<double,2>& theta_v0,
                                  const std::array<double,2>& theta_v_step = {0.01, 0.01},
                                  const double& var_terminating_lim = 1e-5);

double Sigma2Update(const arma::mat& Omega_sum,
                    const int n,
                    const int T);

arma::vec BetaUpdate(const arma::cube& Xbeta,
                     const arma::mat& y,
                     const arma::mat& z,
                     double alpha,
                     const arma::mat& Xz,
                     const arma::mat& inv_mXbeta_sum);

