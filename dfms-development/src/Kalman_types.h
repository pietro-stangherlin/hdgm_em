#pragma once

#include<armadillo>
#include<optional>

struct KalmanFilterResult {
  arma::mat F;         // Filtered state means
  arma::cube P;        // Filtered state covariances
  arma::mat F_pred;    // Predicted state means
  arma::cube P_pred;   // Predicted state covariances
  arma::mat K_last; // Kalman gain value for the last iteration (used in the smoother first step)
  arma::mat C_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  double loglik;  // Default to NA
};


struct KalmanSmootherResult {
  arma::mat F_smooth;       // Smoothed state means
  arma::cube P_smooth;      // Smoothed state covariances
  arma::cube Lag_one_cov_smooth; // Lag one smoothed covariances
  arma::colvec F_smooth_0;  // smoothed initial state mean
  arma::mat P_smooth_0;     // smoothed initial state covariance
  bool has_initial_state;  // Flag to indicate if those were set
};


struct KalmanFilterSmootherResult {
  arma::mat F;             // Filtered means
  arma::cube P;            // Filtered covariances

  arma::mat F_pred;        // Predicted means
  arma::cube P_pred;       // Predicted covariances

  arma::mat F_smooth;      // Smoothed means
  arma::cube P_smooth;     // Smoothed covariances

  arma::colvec F_smooth_0;  // Smoothed initial state
  arma::mat P_smooth_0;

  arma::cube Lag_one_cov_smooth;   // Lag one state covariances
  std::optional<double> loglik;  // Optional log-likelihood
};
