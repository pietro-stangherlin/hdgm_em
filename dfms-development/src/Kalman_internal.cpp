#include <optional>
#include <armadillo>
#include <iostream>
#include <cstdlib> // For atoi

#include"Kalman_internal.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Compile with: (delete the //)
//g++ Kalman_internal.cpp -o Kalman_internal.exe -O2 -std=c++17 \
//-larmadillo -llapack -lblas -lgfortran -lquadmath             \
// -static-libgcc -static-libstdc++

// one liner compile
// g++ Kalman_internal.cpp -o Kalman_internal.exe -O2 -std=c++17 -larmadillo -llapack -lblas -lgfortran -lquadmath -static-libgcc -static-libstdc++


// Change from github code: swap data matrix columns and rows definition
// so each observation is read by column (much more efficient in Armadillo)
// instead of by row


// Kalman Filter
// for parameters description see Kalman_types.h
KalmanFilterResult SKF_cpp(KalmanFilterInput inp) {

  const int n = inp.X.n_rows;
  const int T = inp.X.n_cols;
  const int rp = inp.A.n_rows;
  int n_c;


  // In internal code factors are Z (instead of F) and factor covariance V (instead of P),
  // to avoid confusion between the matrices and their predicted (p) and filtered (f) states.
  // Additionally the results matrices for all time periods have a T in the name.

  double loglik = inp.retLL ? 0.0 : std::nan(""), dn = 0, detS;
  if (inp.retLL) dn = n * std::log(2.0 * M_PI);

  arma::colvec Zp, Zf = inp.F_0, et, xt;
  arma::mat K, Vp, Vf = inp.P_0, S, VCt;

  // Predicted state mean and covariance
  arma::mat ZTp(rp, T, arma::fill::zeros);
  arma::cube VTp(rp, rp, T, arma::fill::zeros);

  // Filtered state mean and covariance
  arma::mat ZTf(rp, T, arma::fill::zeros);
  arma::cube VTf(rp, rp, T, arma::fill::zeros);

  // Handling missing values in the filter
  arma::mat Ci, Ri;
  arma::uvec nmiss, arow = arma::find_finite(inp.A.row(0));
  if (arow.n_elem == 0) {
    throw std::runtime_error("Missing first row of transition matrix");
  }

  for (int i = 0; i < T; ++i) {


    // Run a prediction
    Zp = inp.A * Zf;
    Vp = inp.A * Vf * inp.A.t() + inp.Q;
    Vp += Vp.t(); // Ensure symmetry
    Vp *= 0.5;

    // If missing observations are present at some timepoints, exclude the
    // appropriate matrix slices from the filtering procedure.
    xt = inp.X.col(i);
    nmiss = find_finite(xt);
    n_c = nmiss.n_elem;
    if(n_c > 0) {
      if(n_c == n) {
        Ci = inp.C;
        Ri = inp.R;
      } else {
        Ci = inp.C.submat(nmiss, arow);
        Ri = inp.R.submat(nmiss, nmiss);
        xt = xt.elem(nmiss);
      }


      // Intermediate results
      VCt = Vp * Ci.t();
      S = inv(Ci * VCt + Ri); // .i();


      // Prediction error
      et = xt - Ci * Zp;
      // Kalman gain
      K = VCt * S;
      // Updated state estimate
      Zf = Zp + K * et;
      // Updated state covariance estimate
      Vf = Vp - K * Ci * Vp;
      Vf += Vf.t(); // Ensure symmetry
      Vf *= 0.5;

      // Compute likelihood. Skip this part if S is not positive definite.
      if(inp.retLL) {
        detS = det(S);
        if(detS > 0) loglik += log(detS) - arma::as_scalar(et.t() * S * et) - dn;

      }

    } else { // If all missing: just prediction.
      Zf = Zp;
      Vf = Vp;
    }


    // Store predicted and filtered data needed for smoothing
    ZTp.col(i) = Zp;
    VTp.slice(i) = Vp;
    ZTf.col(i) = Zf;
    VTf.slice(i) = Vf;

  }


  if(inp.retLL) loglik *= 0.5;

  return KalmanFilterResult{
    .F = ZTf, // filtered states
    .P = VTf, // filtered states Variances
    .F_pred = ZTp, // predicted states
    .P_pred = VTp, // predicted states variances
    .K_last = K, // Kalman gain last observation
    .C_last = Ci, // Observation matrix submatrix for non-missing (used in the smoother first step)
    .nc_last = n_c,
    .loglik = loglik
  };
}

// Kalman Smoother
// for parameters description see Kalman_types.h
KalmanSmootherResult FIS_cpp(KalmanSmootherInput inp) {
  const int T = inp.ZTf.n_cols;
  const int rp = inp.A.n_rows;

  arma::mat Vf = inp.VTf.slice(T-1);
  arma::mat Vp = inp.VTp.slice(T-1);
  arma::mat K; // Kalman gain last observation

  // Kalman smoothing
  arma::mat ZsT(rp, T, arma::fill::zeros);
  arma::cube VsT(rp, rp, T, arma::fill::zeros);
  arma::cube VVsT(rp, rp, T, arma::fill::zeros);


  // populate last smoothed values
  ZsT.col(T-1) = inp.ZTf.col(T-1); // last smoothed state = filtered state
  VsT.slice(T-1) = inp.VTf.slice(T-1); // last smoothed state cov = filtered state cov


  arma::mat Ji, Jimt;
  arma::mat At = inp.A.t();

  K = (inp.nc_last == 0) ? arma::mat(rp, rp, arma::fill::zeros) : inp.K_last * inp.C_last;

  VVsT.slice(T-1) = (arma::eye(rp,rp) - K) * inp.A * inp.VTf.slice(T-2);

  // Smoothed state variable and covariance
  for (int t = T - 2; t >= 0; --t) {
    arma::mat Vf = inp.VTf.slice(t);
    arma::mat Vp = inp.VTp.slice(t+1);
    Ji = Vf * At * inv_sympd(Vp);

    arma::mat Jimt = Ji.t();

    ZsT.col(t) = inp.ZTf.col(t) + Ji * (ZsT.col(t+1) - inp.ZTp.col(t+1));
    VsT.slice(t) = Vf + Ji * (VsT.slice(t+1) - Vp) * Jimt;

    // Cov(Z_t, Z_t-1): Needed for EM
    if (t > 0) {
      Jimt = inp.VTf.slice(t-1) * At * inv_sympd(inp.VTp.slice(t));
      VVsT.slice(t) = inp.VTf.slice(t) * Jimt +
        Ji * (VVsT.slice(t+1) - inp.A * inp.VTf.slice(t)) * Jimt;
    }
  }

  // Smoothing t = 0
  Vp = inp.VTp.slice(0);
  Jimt = inp.P_0 * At * inv_sympd(Vp);
  VVsT.slice(0) = inp.VTf.slice(0) * Jimt.t() +
    Ji * (VVsT.slice(1) - inp.A * inp.VTf.slice(0)) * Jimt.t();

  // Initial smoothed values
  arma::colvec F_smooth_0 = inp.F_0 + Jimt * (ZsT.col(0) - inp.ZTp.col(0));
  arma::mat P_smooth_0 = inp.P_0 + Jimt * (VsT.slice(0) - Vp) * Jimt.t();


  return KalmanSmootherResult{
  .F_smooth = ZsT,
  .P_smooth = VsT,
  .Lag_one_cov_smooth = VVsT,
  .F_smooth_0 = F_smooth_0,
  .P_smooth_0 = P_smooth_0
  };
}

// Kalman Filter and Smoother
// Only Kalman Smoother ouptput is returned
// for parameters description see Kalman_types.h
KalmanSmootherResult SKFS_cpp(KalmanFilterInput inp) {

  KalmanFilterResult kf = SKF_cpp(inp);

  KalmanSmootherInput ksmin = {
  .A = inp.A,
  .ZTf = kf.F,
  .ZTp = kf.F_pred,
  .VTf = kf.P,
  .VTp = kf.P_pred,
  .K_last = kf.K_last,
  .C_last = kf.C_last,
  .F_0 = inp.F_0,
  .P_0 = inp.P_0,
  .nc_last = kf.nc_last,
  };

  std::cout << "Kalman Filter finished\n";

  KalmanSmootherResult ks = FIS_cpp(ksmin);

  std::cout << "Kalman Smoother finished\n";

  return ks;
}


// Add testing


int main(int argc, char* argv[]) {
  // Default values
  int T = 100;
  int n = 5;
  int m = 2;

  // If arguments are provided, override the defaults
  if (argc > 1) T = std::atoi(argv[1]);
  if (argc > 2) n = std::atoi(argv[2]);
  if (argc > 3) m = std::atoi(argv[3]);

  std::cout << "T = " << T << ", n = " << n << ", m = " << m << std::endl;
  // Simulate small input matrices for testing
  KalmanFilterInput test_kfin{
  .X = arma::randn(n, T), // observations matrix
  .A = arma::randn(m, m), // Transition matrix
  .C = arma::eye(n, m), // observation matrix
  .Q = arma::eye(m, m) * 0.1, // state covariance error matrix
  .R = arma::eye(n, n), // observation error covariance matrix
  .F_0 = arma::zeros(m), // first state
  .P_0 = arma::eye(m, m), // first state covariance
  .retLL = true};

  bool compute_llik = true;

  std::cout << "Kalman Filter: \n";
  KalmanFilterResult resf = SKF_cpp(test_kfin);
  std::cout << "Test passed. Log-likelihood: " << resf.loglik;

  std::cout << "\n";

  KalmanSmootherInput test_ksmin = {
    .A = test_kfin.A,
    .ZTf = resf.F,
    .ZTp = resf.F_pred,
    .VTf = resf.P,
    .VTp = resf.P_pred,
    .K_last = resf.K_last,
    .C_last = resf.C_last,
    .F_0 = test_kfin.F_0,
    .P_0 = test_kfin.P_0,
    .nc_last = resf.nc_last,
  };

  std::cout << "Kalman Smoother: \n";
  KalmanSmootherResult ressm = FIS_cpp(test_ksmin);
  std::cout << "Test passed\n";

  std::cout << "Kalman Filter and Smoother: \n";
  KalmanSmootherResult reskfsm = SKFS_cpp(test_kfin);
  std::cout << "Test passed\n";


  return 0;
}


