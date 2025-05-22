#pragma once

#include"Kalman_types.h"

KalmanFilterResult SKF_cpp(const KalmanFilterInput& kf_inp);


// DEBUG version
double SKFExplicitInput_cpp(const arma::mat& X, // observation matrix: each observation is a column
                                        const arma::mat& A, // state transition matrix
                                        const arma::mat& C, // observation matrix
                                        const arma::mat& Q, // state covariance error matrix
                                        const arma::mat& R, // observation error covariance matrix
                                        const arma::vec& F_0, // initial state
                                        const arma::mat& P_0, // initial state covariance
                                        bool retLL);


KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp);

KalmanSmootherResult SKFS_cpp(const KalmanFilterInput& kfsm_inp);
