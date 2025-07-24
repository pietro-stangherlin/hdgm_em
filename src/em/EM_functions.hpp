#pragma once

#include <RcppArmadillo.h>

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
arma::mat OmegaSumUpdate_core(const arma::mat & mY_fixed_res,
                                         const arma::mat & Zt,
                                         const arma::mat & mXz,
                                         CovStore & cPsmt,
                                         double alpha);
