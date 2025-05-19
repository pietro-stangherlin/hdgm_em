#pragma once

#include"Kalman_types.h"


KalmanFilterResult SKF_cpp(arma::mat X,
                           arma::mat A,
                           arma::mat C,
                           arma::mat Q,
                           arma::mat R,
                           arma::colvec F_0,
                           arma::mat P_0,
                           bool retLL = false);

KalmanSmootherResult FIS_cpp(const arma::mat& A,
                             const arma::mat& ZTf,
                             const arma::mat& ZTp,
                             const arma::cube& VTf,
                             const arma::cube& VTp,
                             const arma::mat& K_last,
                             const arma::mat& C_last,
                             const arma::colvec& F_0,
                             const arma::mat& P_0);

KalmanFilterSmootherResult SKFS_cpp(const arma::mat& X,
                                    const arma::mat& A,
                                    const arma::mat& C,
                                    const arma::mat& Q,
                                    const arma::mat& R,
                                    const arma::colvec& F_0,
                                    const arma::mat& P_0,
                                    bool retLL = false);
