#include <armadillo>
#include <stdio.h>

#include "kalman/Kalman_internal.h"
#include "em/EM_functions.h"
#include "em/EM_algorithm.h"


// assuming no missing observations and no matrix permutations
// temporarely return integer, then return all parameters
EMOutput EMHDGM(EMInput em_in) {

  // Setup
  bool is_fixed_effect = em_in.Xbeta.has_value();

  int q = em_in.y.n_rows; // observation and state vector length
  int T = em_in.y.n_cols;

  int p = em_in.beta0.n_elem; // fixed effect vector length

  double alpha_temp = em_in.alpha0;
  double theta_temp = em_in.theta0;
  double g_temp = em_in.g0;
  double sigma2_temp = em_in.sigma20;

  arma::vec beta_temp = em_in.beta0;


  // NOTE: parameter dimension has to be changed if the parameters space change
  // also, zero is not a perfect initialization value
  arma::mat par_history = arma::mat(4, em_in.max_iter + 1);
  arma::mat beta_history = arma::mat(p, em_in.max_iter + 1);

  par_history.col(0) = arma::vec({alpha_temp, theta_temp, g_temp, sigma2_temp});
  beta_history.col(0) = beta_temp;


  //arma::mat param_history(max_iter + 1, 4, arma::fill::zeros);
  //param_history.row(0) = arma::rowvec({alpha_temp, theta_temp, g_temp, sigma2_temp});

  //arma::mat beta_iter_history;
  //if (is_fixed_effect)
    //beta_iter_history.set_size(max_iter + 1, p);
  //if (is_fixed_effect)
    //beta_iter_history.row(0) = beta_temp.t();

  arma::vec z0 = em_in.z0_in.has_value() ? *em_in.z0_in : arma::vec(q, arma::fill::zeros);
  arma::mat P0 = em_in.P0_in.has_value() ? *em_in.P0_in : arma::eye(q, q);
  arma::mat Xz = arma::eye(q, q);  // Transfer matrix

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

  // EM iterations
  for (int iter = 1; iter < em_in.max_iter + 1; ++iter) {

    //if (em_in.verbose)
      //std::cout << "Iteration " << iter << std::endl;

    // Subtract fixed effects
    arma::mat y_res = em_in.y;

    if (is_fixed_effect) {
      for (int t = 0; t < T; ++t) {
        y_res.col(t) -= (*em_in.Xbeta).slice(t) * beta_temp;
      }
    }

    //std::cout << "[DEBUG] y_res (fixed effect done) " <<std::endl;

    // Update Q
    arma::mat Q_temp = exp(-theta_temp * em_in.dist_matrix);
    //std::cout << "[DEBUG] Q_temp matrix updated " << std::endl;

    ///////////////////////
    // Kalman Smoother pass
    ///////////////////////

    KalmanFilterInput kfin{
      .X = y_res, // observations matrix
      .A = g_temp * arma::eye(q, q), // Transition matrix
      .C = alpha_temp * arma::eye(q, q), // observation matrix
      .Q = Q_temp, // state covariance error matrix
      .R = sigma2_temp * arma::eye(q, q), // observation error covariance matrix
      .F_0 = z0, // first state
      .P_0 = P0, // first state covariance
      .retLL = true};

    KalmanSmootherResult ksm_res = SKFS_cpp(kfin);

    //std::cout << "KalmanSmootherPassDone" << std::endl;

    // TO DO: review assignemt
    arma::mat& z_smooth = ksm_res.F_smooth;
    arma::cube& z_smooth_var = ksm_res.P_smooth;
    arma::mat& z0_smooth = ksm_res.F_smooth_0;
    arma::mat& P0_smooth = ksm_res.P_smooth_0;
    arma::cube& lag1_cov =  ksm_res.Lag_one_cov_smooth;

    // S matrices
    arma::mat S00 = ComputeS00(z_smooth , z_smooth_var, z0_smooth, P0_smooth);
    arma::mat S11 = ComputeS11(z_smooth, z_smooth_var, S00, z0_smooth, P0_smooth);
    arma::mat S10 = ComputeS10(z_smooth, lag1_cov, z0_smooth);

    //////////////////////////
    // EM parameters updates
    /////////////////////////

    // Omega Update
    arma::mat omega_sum_temp = OmegaSumUpdate(y_res, // residual minus fixed effect
                                              z_smooth,
                                              Xz,
                                              z_smooth_var,
                                              alpha_temp);


    // Sigma2 update
    sigma2_temp = Sigma2Update(omega_sum_temp, q, T);

    // Alpha update
    alpha_temp = AlphaUpdate(em_in.y, z_smooth, Xz, z_smooth_var);

    // Beta update
    if (is_fixed_effect) {
      const arma::cube& Xbeta_val = *(em_in.Xbeta);
      beta_temp = BetaUpdate(Xbeta_val,
                             em_in.y,
                             z_smooth,
                             alpha_temp,
                             Xz,
                             m_inv_mXbeta_sum);

      // beta_iter_history.row(iter + 1) = beta_temp.t();
    }

    // Theta update (optimization)
    theta_temp = ThetaUpdate(em_in.dist_matrix, g_temp,
                             S00, S10, S11,
                             theta_temp,
                             T, em_in.sigma2_lower, em_in.sigma2_upper);

    // g update
    g_temp = gUpdate(S00, S10);

  // Update parameters history
  par_history.col(iter) = arma::vec({alpha_temp, theta_temp, g_temp, sigma2_temp});
  beta_history.col(iter) = beta_temp;
  }




  // Assemble return

  // if (is_fixed_effect)
    //out["beta_iter_history"] = beta_iter_history;

  return EMOutput{.par_history = par_history,
                  .beta_history = beta_history};
};
