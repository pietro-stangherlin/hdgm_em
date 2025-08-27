#pragma once
#include <RcppArmadillo.h>


// Input and Output structures

// ---------------------- Input -------------------------- //

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

// NOTE: for time varying case
// the observation matrix A can change with time
// more specifically, we assume it to be a block diagonal matrix
//|upper_block_t 0          |
//|0             lower_block|
// where the upper block is time varying
// and the lower block is time fixed
// to not waste storage on the matrix zeros
// we pass the two blocks (cube of matrices for the upper and matrix for the lower)
// as separate inputs

struct KalmanFilterInputTimeVaryingObsMatr {
  const arma::mat& Y; // matrix with observation: each observation is a column
  const arma::mat& Phi; // state transition matrix
  const arma::cube& A_upper; // observation matrix upper block time varying
  const arma::mat& A_lower; // observation matrix lower block time fixed
  const arma::mat& Q; // state covariance error matrix
  const arma::mat& R; // observation error covariance matrix
  const arma::vec& x_0; // initial state
  const arma::mat& P_0; // initial state covariance
  bool retLL;
};


template <typename PType>
struct KalmanSmootherInputT {
  const arma::mat& Phi; // state transition matrix
  const arma::mat& xf; // Filtered state means
  const arma::mat& xp; // Predicted state means
  const PType& Pf; // Filtered state covariances
  const PType& Pp; // Predicted state covariances
  const arma::mat& K_last; // Kalman gain last observation
  const arma::mat& A_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  const arma::colvec& x_0; // Initial state vector (rp x 1)
  const arma::mat& P_0; // Initial state covariance (rp x rp)
  const int nc_last; // number of non missing elements in the last observation
};

using KalmanSmootherInput = KalmanSmootherInputT<arma::cube>;
using KalmanSmootherInputMat = KalmanSmootherInputT<arma::mat>;

// ---------------------- Output -------------------------- //

template <typename PType>
struct KalmanFilterResultT {
  arma::mat xf;         // Filtered state means
  PType Pf;        // Filtered state covariances
  arma::mat xp;    // Predicted state means
  PType Pp;   // Predicted state covariances
  arma::mat K_last; // Kalman gain value for the last iteration (used in the smoother first step)
  arma::mat A_last; // Observation matrix submatrix for non-missing (used in the smoother first step)
  int nc_last;
  double loglik;  // Default to NA
};

using KalmanFilterResult = KalmanFilterResultT<arma::cube>;
using KalmanFilterResultMat = KalmanFilterResultT<arma::mat>;


template <typename PType>
struct KalmanSmootherResultT {
  arma::mat x_smoothed;
  PType P_smoothed;
  PType Lag_one_cov_smoothed;
  arma::colvec x0_smoothed;
  arma::mat P0_smoothed;
};

using KalmanSmootherResult = KalmanSmootherResultT<arma::cube>;
using KalmanSmootherResultMat = KalmanSmootherResultT<arma::mat>;

// while doing Kalman Filter and Smoother in one pass
// return smoothed quantities and likelihood from the filter
struct KalmanSmootherLlikResult : KalmanSmootherResult {
  double loglik; // likelihood from Kalman Filter

  // constructor
  KalmanSmootherLlikResult(const KalmanSmootherResult& ksm_res, double loglik):
    KalmanSmootherResult(ksm_res), loglik(loglik){}
};

struct KalmanSmootherLlikResultMat : KalmanSmootherResultMat {
  double loglik; // likelihood from Kalman Filter

  // constructor
  KalmanSmootherLlikResultMat(const KalmanSmootherResultMat& ksm_res, double loglik):
    KalmanSmootherResultMat(ksm_res), loglik(loglik){}
};

