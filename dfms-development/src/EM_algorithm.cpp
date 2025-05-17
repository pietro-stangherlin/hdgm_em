#include <RcppArmadillo.h>
#include "KalmanFiltering.h"
#include "helper.h"
#include <stdio.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;
using namespace std;


// assuming no missing observations and no matrix permutations

/**
* @description EM update for scale parameter of influence of state variables
* one observation vector (eq. 4)
* (common to each state)
*
* @param mY (matrix): (T x 1) matrix of observed vector, NA allowed
* (a sorting is assumed, example by spatial locations)
* @param mZ (matrix): (T x s) matrix of smoothed state vectors
* @param cXbeta (array) (q x p x T) array of fixed effects covariates matrices,
*  each of those is accessed by Xbeta[,t]
* @param vbeta (vector) (p x 1) matrix (i.e. a vector) of fixed effects coef,
* does NOT change with time
* @param mXz (array) (s x s) non scaled transfer matrix (assumed constant in time
* (the complete transfer matrix is scaled by alpha)
* @param cPsm (array): (s x s x T) array of smoothed state variance matrices,
*  each of those is accessed by cPsm[,t]
*/

// [[Rcpp::export]]
float AlphaUpdate(arma::mat & mY, arma::mat & mZ,
                  arma::cube cXbeta, arma::mat mXz,
                  arma::vec vbeta, arma::cube cPsm){

}



// [[Rcpp::export]]
Rcpp::List Estep(arma::mat X, arma::mat A, arma::mat C, arma::mat Q,
                 arma::mat R, arma::colvec F_0, arma::mat P_0) {

  const unsigned int T = X.n_rows;
  const unsigned int n = X.n_cols;
  const unsigned int rp = A.n_rows;

  // Run Kalman filter and Smoother
  List ks = SKFS(X, A, C, Q, R, F_0, P_0, true);
  double loglik = as<double>(ks["loglik"]);
  mat Fs = as<mat>(ks["F_smooth"]);
  cube Psmooth = array2cube(as<NumericVector>(ks["P_smooth"]));
  cube Wsmooth = array2cube(as<NumericVector>(ks["PPm_smooth"]));

  // Run computations and return all estimates
  mat delta(n, rp); delta.zeros();
  mat gamma(rp, rp); gamma.zeros();
  mat beta(rp, rp); beta.zeros();

  // For E-step purposes it is sufficient to set missing observations
  // to being 0.
  X(find_nonfinite(X)).zeros();

  for (unsigned int t=0; t<T; ++t) {
    delta += X.row(t).t() * Fs.row(t);
    gamma += Fs.row(t).t() * Fs.row(t) + Psmooth.slice(t);
    if (t > 0) {
      beta += Fs.row(t).t() * Fs.row(t-1) + Wsmooth.slice(t);
    }
  }

  mat gamma1 = gamma - Fs.row(T-1).t() * Fs.row(T-1) - Psmooth.slice(T-1);
  mat gamma2 = gamma - Fs.row(0).t() * Fs.row(0) - Psmooth.slice(0);
  colvec F1 = Fs.row(0).t();
  mat P1 = Psmooth.slice(0);

  return Rcpp::List::create(Rcpp::Named("beta") = beta,
                            Rcpp::Named("gamma") = gamma,
                            Rcpp::Named("delta") = delta,
                            Rcpp::Named("gamma1") = gamma1,
                            Rcpp::Named("gamma2") = gamma2,
                            Rcpp::Named("F_0") = F1,
                            Rcpp::Named("P_0") = P1,
                            Rcpp::Named("loglik") = loglik);
                            // Rcpp::Named("Fs") = ks["Fs"]);
}

