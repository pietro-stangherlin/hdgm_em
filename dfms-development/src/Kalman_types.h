#pragma once

#include<armadillo>
#include<optional>


// Input and Output structures

// ---------------------- Input -------------------------- //

// DOUBT: can I use the inputs passed as reference?
// maybe I have to use pointers

struct KalmanFilterInput {
  const arma::mat& X; // observation matrix: each observation is a column
  const arma::mat& A; // state transition matrix
  const arma::mat& C; // observation matrix
  const arma::mat& Q; // state covariance error matrix
  const arma::mat& R; // observation error covariance matrix
  const arma::colvec& F_0; // initial state
  const arma::mat& P_0; // initial state covariance
  bool retLL;
};

// USED ONLY IN RCPP wrapper
// test data structure, only used in trying to solve
// the Rcpp memory bug
struct KalmanFilterInputByValue {
  const arma::mat X; // observation matrix: each observation is a column
  const arma::mat A; // state transition matrix
  const arma::mat C; // observation matrix
  const arma::mat Q; // state covariance error matrix
  const arma::mat R; // observation error covariance matrix
  const arma::colvec F_0; // initial state
  const arma::mat P_0; // initial state covariance
  bool retLL;
};

struct KalmanSmootherInput {
  const arma::mat& A; // state transition matrix
  const arma::mat& ZTf; // State estimates
  const arma::mat& ZTp; // State predicted estimates
  const arma::cube& VTf; // Variance estimates
  const arma::cube& VTp; // Predicted variance estimates
  const arma::mat& K_last; // Kalman gain last observation
  const arma::mat& C_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  const arma::colvec& F_0; // Initial state vector (rp x 1)
  const arma::mat& P_0; // Initial state covariance (rp x rp)
  const int nc_last; // number of non missing elements in the last observation
};

// ---------------------- Output -------------------------- //

struct KalmanFilterResult {
  arma::mat F;         // Filtered state means
  arma::cube P;        // Filtered state covariances
  arma::mat F_pred;    // Predicted state means
  arma::cube P_pred;   // Predicted state covariances
  arma::mat K_last; // Kalman gain value for the last iteration (used in the smoother first step)
  arma::mat C_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  int nc_last;
  double loglik;  // Default to NA
};


struct KalmanSmootherResult {
  arma::mat F_smooth;       // Smoothed state means
  arma::cube P_smooth;      // Smoothed state covariances
  arma::cube Lag_one_cov_smooth; // Lag one smoothed covariances
  arma::colvec F_smooth_0;  // smoothed initial state mean
  arma::mat P_smooth_0;     // smoothed initial state covariance
};

