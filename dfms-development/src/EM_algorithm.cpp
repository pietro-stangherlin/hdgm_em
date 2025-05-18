#include <RcppArmadillo.h>
#include <stdio.h>
#include<"EM_functions.cpp">

using namespace std;


// assuming no missing observations and no matrix permutations

// [[Rcpp::export]]
Rcpp::List EMHDGM(const arma::mat& y,
                  const arma::mat& dist_matrix,
                  double alpha0,
                  const arma::vec& beta0,
                  double theta0,
                  double g0,
                  double sigma20,
                  Nullable<arma::cube> Xbeta = R_NilValue,
                  Nullable<arma::vec> z0_in = R_NilValue,
                  Nullable<arma::mat> P0_in = R_NilValue,
                  int max_iter = 10,
                  double sigma2_upper = 10.0,
                  bool sigma2_do_plot_objective = false,
                  bool verbose = true) {

  // Setup
  bool is_fixed_effect = Xbeta.isNotNull();
  int q = y.n_rows;
  int T = y.n_cols;
  int p = beta0.n_elem;

  arma::vec beta_temp = beta0;
  double alpha_temp = alpha0;
  double theta_temp = theta0;
  double g_temp = g0;
  double sigma2_temp = sigma20;

  arma::mat param_history(max_iter + 1, 4, arma::fill::zeros);
  param_history.row(0) = arma::rowvec({alpha_temp, theta_temp, g_temp, sigma2_temp});

  arma::mat beta_iter_history;
  if (is_fixed_effect)
    beta_iter_history.set_size(max_iter + 1, p);
  if (is_fixed_effect)
    beta_iter_history.row(0) = beta_temp.t();

  arma::vec z0 = z0_in.isNotNull() ? as<arma::vec>(z0_in) : arma::vec(q, arma::fill::zeros);
  arma::mat P0 = P0_in.isNotNull() ? as<arma::mat>(P0_in) : arma::eye(q, q);
  arma::mat Xz = arma::eye(q, q);  // Transfer matrix

  arma::mat mXbeta_sum;
  arma::mat m_inv_mXbeta_sum;

  // Precompute fixed-effects sum
  if (is_fixed_effect) {
    arma::cube Xb = as<arma::cube>(Xbeta);
    mXbeta_sum.zeros(p, p);
    for (int t = 0; t < T; ++t) {
      arma::mat Xt = Xb.slice(t);
      mXbeta_sum += Xt.t() * Xt;
    }
    m_inv_mXbeta_sum = arma::inv_sympd(mXbeta_sum);
  }

  // EM iterations
  for (int iter = 0; iter < max_iter; ++iter) {

    if (verbose)
      Rcout << "Iteration " << iter << std::endl;

    // Subtract fixed effects
    arma::mat y_res = y;
    if (is_fixed_effect) {
      arma::cube Xb = as<arma::cube>(Xbeta);
      for (int t = 0; t < T; ++t) {
        y_res.col(t) -= Xb.slice(t) * beta_temp;
      }
    }

    // Update Q
    arma::mat Q_temp = exp(-theta_temp * dist_matrix);

    // Kalman Smoother pass
    List ksm_res = SKFS(y_res.t(),             // X
                        g_temp * arma::eye(q, q), // A
                        alpha_temp * Xz,       // C
                        Q_temp,                // Q
                        sigma2_temp * arma::eye(q, q), // R
                        z0,                    // F_0
                        P0);                   // P_0

    arma::mat z_smooth = as<arma::mat>(ksm_res["F_smooth"]).t();
    arma::cube z_smooth_var = as<arma::cube>(ksm_res["P_smooth"]);
    arma::mat z0_smooth = as<arma::mat>(ksm_res["F_smooth_0"]).t();
    arma::mat P0_smooth = as<arma::mat>(ksm_res["P_smooth_0"]);
    arma::cube lag1_cov = as<arma::cube>(ksm_res["PPm_smooth"]);

    // S matrices
    arma::mat S00 = ComputeS00(z_smooth, z_smooth_var, z0_smooth, P0_smooth);
    arma::mat S11 = ComputeS11(z_smooth, z_smooth_var, S00, z0_smooth, P0_smooth);
    arma::mat S10 = ComputeS10(z_smooth, lag1_cov, z0_smooth);

    // Sigma2 update
    sigma2_temp = Sigma2Update(y, z_smooth, alpha_temp, Xz);

    // Alpha update
    alpha_temp = AlphaUpdate(y, z_smooth, Xbeta, beta_temp, Xz, z_smooth_var);

    // Beta update
    if (is_fixed_effect) {
      beta_temp = BetaUpdate(as<arma::cube>(Xbeta), y, z_smooth, alpha_temp, Xz, m_inv_mXbeta_sum);
      beta_iter_history.row(iter + 1) = beta_temp.t();
    }

    // Theta update (optimization)
    theta_temp = ThetaUpdate(dist_matrix, g_temp, S00, S10, S11, theta_temp,
                             T, sigma2_upper, sigma2_do_plot_objective);

    // g update
    g_temp = gUpdate(S00, S10);

    // Track parameters
    param_history.row(iter + 1) = arma::rowvec({alpha_temp, theta_temp, g_temp, sigma2_temp});
  }

  // Assemble return
  List out = List::create(
    Named("alpha") = alpha_temp,
    Named("beta") = beta_temp,
    Named("theta") = theta_temp,
    Named("g") = g_temp,
    Named("sigma2") = sigma2_temp,
    Named("iter_history") = param_history
  );

  if (is_fixed_effect)
    out["beta_iter_history"] = beta_iter_history;

  return out;
}
