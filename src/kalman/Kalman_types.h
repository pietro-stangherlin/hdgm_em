#pragma once
#include <RcppArmadillo.h>
#include <optional>


// Input and Output structures

// ---------------------- Input -------------------------- //

// DOUBT: can I use the inputs passed as reference?
// maybe I have to use pointers

struct KalmanFilterInput {
  const arma::mat& Y; // matrix with observation: each observation is a column
  const arma::mat& Phi; // state transition matrix
  const arma::mat& A; // observation matrix
  const arma::mat& Q; // state covariance error matrix
  const arma::mat& R; // observation error covariance matrix
  const arma::vec& x_0; // initial state
  const arma::mat& P_0; // initial state covariance
  bool retLL;

};


struct KalmanSmootherInput {
  const arma::mat& Phi; // state transition matrix
  const arma::mat& xf; // Filtered state means
  const arma::mat& xp; // Predicted state means
  const arma::cube& Pf; // Filtered state covariances
  const arma::cube& Pp; // Predicted state covariances
  const arma::mat& K_last; // Kalman gain last observation
  const arma::mat& A_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  const arma::colvec& x_0; // Initial state vector (rp x 1)
  const arma::mat& P_0; // Initial state covariance (rp x rp)
  const int nc_last; // number of non missing elements in the last observation
};

// ---------------------- Output -------------------------- //

struct KalmanFilterResult {
  arma::mat xf;         // Filtered state means
  arma::cube Pf;        // Filtered state covariances
  arma::mat xp;    // Predicted state means
  arma::cube Pp;   // Predicted state covariances
  arma::mat K_last; // Kalman gain value for the last iteration (used in the smoother first step)
  arma::mat A_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  int nc_last;
  double loglik;  // Default to NA
};


struct KalmanSmootherResult {
  arma::mat x_smoothed;       // Smoothed state means
  arma::cube P_smoothed;      // Smoothed state covariances
  arma::cube Lag_one_cov_smoothed; // Lag one smoothed covariances
  arma::colvec x0_smoothed;  // smoothed initial state mean
  arma::mat P0_smoothed;     // smoothed initial state covariance
};

