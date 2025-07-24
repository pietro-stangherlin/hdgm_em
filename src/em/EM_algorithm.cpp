#include <RcppArmadillo.h>
#include <stdio.h>
#include <limits>

#include "../kalman/Kalman_internal.h"
#include "EM_functions.h"
#include "EM_functions.hpp"
#include "EM_algorithm.h"
#include "../utils/covariances.h"


constexpr double LOWEST_DOUBLE = std::numeric_limits<double>::lowest();


// -------------------- Unstructured Case --------------------- //
// from Parameter estimation for linear dynamical systems.
// Ghahramani, Z. & Hinton, G. E. Technical Report Technical Report CRG-TR-96-2,
// University of Totronto, Dept. of Computer Science, 1996.
EMOutputUnstructured UnstructuredEM_cpp(EMInputUnstructured em_in){

  // ------------------ setup ------------- //
  int p = em_in.x0_in.n_rows; // state vector dimension
  int q = em_in.y.n_rows; // observation vector dimension
  int T = em_in.y.n_cols; // number of observations

  arma::mat y, Phi, A, Q, R, P0_smooth;
  arma::vec x0_smooth;
  double llik_prev, llik_next;
  llik_prev = LOWEST_DOUBLE;

  y = em_in.y;
  Phi = em_in.Phi_0;
  A = em_in.A_0;
  Q = em_in.Q_0;
  R = em_in.R_0;

  x0_smooth = em_in.x0_in;
  P0_smooth = em_in.P0_in;

  // once for all
  arma::mat sum_y_yT(q, q, arma::fill::zeros);
  for(int t = 0; t < T; t++){
    sum_y_yT += y.col(t) * y.col(t).t();
  };


  for(int iter = 1; iter < em_in.max_iter + 1; ++iter){

    std::cout << "iter" << iter << std::endl;


    ///////////////////////
    // Kalman Smoother pass
    ///////////////////////

    KalmanFilterInput kfin{
      .Y = y,
      .Phi = Phi,
      .A = A,
      .Q = Q,
      .R = R,
      .x_0 = x0_smooth,
      .P_0 = P0_smooth,
      .retLL = true};


    KalmanSmootherLlikResult ksm_res = SKFS_cpp(kfin, std::type_identity<arma::cube>{});

    llik_next = ksm_res.loglik;

    if(llik_next < llik_prev){
      std::cout << "WARNING: Log Likelihood decreasing, returning" << std::endl;
      return EMOutputUnstructured{ .Phi = Phi, .A = A, .Q = Q, .R = R};

    }

    llik_prev = llik_next;

    //////////////////////////
    // EM parameters updates
    /////////////////////////

    x0_smooth = ksm_res.x0_smoothed;
    P0_smooth = ksm_res.P0_smoothed;

    // S matrices
    arma::mat S00 = ComputeS00(ksm_res.x_smoothed, ksm_res.P_smoothed, ksm_res.x0_smoothed, ksm_res.P0_smoothed);
    arma::mat S11 = ComputeS11(ksm_res.x_smoothed, ksm_res.P_smoothed, S00, ksm_res.x0_smoothed, ksm_res.P0_smoothed);
    arma::mat S10 = ComputeS10(ksm_res.x_smoothed, ksm_res.Lag_one_cov_smoothed, ksm_res.x0_smoothed);

    arma::mat sum_y_x_smooth(q, p, arma::fill::zeros);
    for(int t = 0; t < T; t++){
      sum_y_x_smooth += y.col(t) * ksm_res.x_smoothed.col(t).t();
    };


    A = sum_y_x_smooth * arma::inv(S11);
    R = (sum_y_yT - A *  sum_y_x_smooth.t()) / T ;
    Phi = S10 * arma::inv(S00);
    Q = (S11 - Phi * S10.t()) / T;

  }


  return EMOutputUnstructured{ .Phi = Phi, .A = A, .Q = Q, .R = R};


};


//  ------------------ Structured Case ------------------------ //
// assuming no missing observations and no matrix permutations
EMOutput EMHDGM_cpp(EMInput em_in) {

  // Setup
  bool is_fixed_effect = em_in.Xbeta.has_value();

  int q = em_in.y.n_rows; // observation and state vector length
  int T = em_in.y.n_cols;

  int p = em_in.beta0.n_elem; // fixed effect vector length

  double alpha_temp = em_in.alpha0;

  std::array<double,2> theta_v_temp = {em_in.theta0, em_in.v0};

  double g_temp = em_in.g0;
  double sigma2_temp = em_in.sigma20;

  arma::vec beta_temp = em_in.beta0;


  // NOTE: parameter dimension has to be changed if the parameters space change
  // also, zero is not a perfect initialization value
  arma::mat par_history = arma::mat(5, em_in.max_iter + 1);
  arma::mat beta_history = arma::mat(p, em_in.max_iter + 1);

  par_history.col(0) = arma::vec({alpha_temp, theta_v_temp[0], theta_v_temp[1],
                  g_temp, sigma2_temp});
  beta_history.col(0) = beta_temp;


  //arma::mat param_history(max_iter + 1, 4, arma::fill::zeros);
  //param_history.row(0) = arma::rowvec({alpha_temp, theta_temp, g_temp, sigma2_temp});

  //arma::mat beta_iter_history;
  //if (is_fixed_effect)
    //beta_iter_history.set_size(max_iter + 1, p);
  //if (is_fixed_effect)
    //beta_iter_history.row(0) = beta_temp.t();

  // at first iteration this is not smoothed
  arma::vec z0_smooth = em_in.z0_in.has_value() ? *em_in.z0_in : arma::vec(q, arma::fill::zeros);
  arma::mat P0_smooth = em_in.P0_in.has_value() ? *em_in.P0_in : arma::eye(q, q);
  arma::mat Xz = arma::eye(q, q);  // unscaled transfer matrix

  // Precompute fixed-effects sum

  // allocate here as a temp scope fix
  arma::mat mXbeta_sum(p, p, arma::fill::zeros);
  arma::mat m_inv_mXbeta_sum;

  if (is_fixed_effect) {

    for (int t = 0; t < T; ++t) {
      mXbeta_sum += (*em_in.Xbeta).slice(t).t() * (*em_in.Xbeta).slice(t);
    }
    m_inv_mXbeta_sum = arma::inv_sympd(mXbeta_sum);
  }

  double llik_prev;
  double llik_next;

  llik_prev = LOWEST_DOUBLE;

  // EM iterations
  for (int iter = 1; iter < em_in.max_iter + 1; ++iter) {

    if (em_in.verbose){
      int remainder;
      remainder = iter % 1;

      if(remainder == 0){
        std::cout << "Iteration " << iter << std::endl;
      };

    };


    // Subtract fixed effects
    arma::mat y_res = em_in.y;

    if (is_fixed_effect) {
      for (int t = 0; t < T; ++t) {
        y_res.col(t) -= (*em_in.Xbeta).slice(t) * beta_temp;
      }
    }

    //std::cout << "[DEBUG] y_res (fixed effect done) " <<std::endl;

    // Update Q
    arma::mat Q_temp = theta_v_temp[1] * ExpCor(em_in.dist_matrix, theta_v_temp[0]);
    //std::cout << "[DEBUG] Q_temp matrix updated " << std::endl;

    ///////////////////////
    // Kalman Smoother pass
    ///////////////////////

    KalmanFilterInput kfin{
      .Y = y_res, // observations matrix
      .Phi = g_temp * arma::eye(q, q), // Transition matrix
      .A = alpha_temp * arma::eye(q, q), // observation matrix
      .Q = Q_temp, // state covariance error matrix
      .R = sigma2_temp * arma::eye(q, q), // observation error covariance matrix
      .x_0 = z0_smooth, // first state
      .P_0 = P0_smooth, // first state covariance
      .retLL = true};

    KalmanSmootherLlikResult ksm_res = SKFS_cpp(kfin, std::type_identity<arma::cube>{});

    llik_next = ksm_res.loglik;

    if(llik_next < llik_prev){
      std::cout << "WARNING: Log Likelihood decreasing, returning" << std::endl;
      return EMOutput{.par_history = par_history,
                      .beta_history = beta_history};

    }

    llik_prev = llik_next;


    //std::cout << "KalmanSmootherPassDone" << std::endl;

    // S matrices
    arma::mat S00 = ComputeS00(ksm_res.x_smoothed, ksm_res.P_smoothed, ksm_res.x0_smoothed, ksm_res.P0_smoothed);
    arma::mat S11 = ComputeS11(ksm_res.x_smoothed, ksm_res.P_smoothed, S00, ksm_res.x0_smoothed, ksm_res.P0_smoothed);
    arma::mat S10 = ComputeS10(ksm_res.x_smoothed, ksm_res.Lag_one_cov_smoothed, ksm_res.x0_smoothed);

    //////////////////////////
    // EM parameters updates
    /////////////////////////

    // Omega Update
    arma::mat omega_sum_temp = OmegaSumUpdate(y_res, // residual minus fixed effect
                                              ksm_res.x_smoothed,
                                              Xz,
                                              ksm_res.P_smoothed,
                                              alpha_temp);


    // Sigma2 update
    sigma2_temp = Sigma2Update(omega_sum_temp, q, T);

    // Alpha update
    alpha_temp = AlphaUpdate(em_in.y, ksm_res.x_smoothed, Xz, ksm_res.P_smoothed);

    // Beta update
    if (is_fixed_effect) {
      const arma::cube& Xbeta_val = *(em_in.Xbeta);
      beta_temp = BetaUpdate(Xbeta_val,
                             em_in.y,
                             ksm_res.x_smoothed,
                             alpha_temp,
                             Xz,
                             m_inv_mXbeta_sum);

    }

    // DEBUG
    // std::cout << "Before ThetaVUpdate" << std::endl;

    // Theta and V update (optimization)
    theta_v_temp = ThetaVUpdate(em_in.dist_matrix, g_temp, T,
                             S00, S10, S11,
                             theta_v_temp,
                             em_in.theta_v_step,
                             em_in.var_terminating_lim);

    // DEBUG
    // std::cout << "After ThetaVUpdate" << std::endl;

    // std::cout << "g_temp: " << g_temp <<  std::endl;

    // g update
    g_temp = gUpdate(S00, S10);

  // Update parameters history
  par_history.col(iter) = arma::vec({alpha_temp, theta_v_temp[0], theta_v_temp[1],
                  g_temp, sigma2_temp});
  beta_history.col(iter) = beta_temp;
  }

  return EMOutput{.par_history = par_history,
                  .beta_history = beta_history};
};
