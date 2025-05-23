#include <armadillo>
#include <stdio.h>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

#include "em/EM_functions.h"
#include "utils/covariances.h"

// Temp file


// covariance specification ----------------------------

// assuming no missing observations and no matrix permutations

/**
 * @description EM update for scale parameter of influence of state variables
 * one observation vector (eq. 4)
 * (common to each state)
 *
 * @param mY (matrix): (T x n) matrix of observed vector
 * with fixed effects predictions subtracted, NA allowed (NOT allowed temporarely)
 * (a sorting is assumed, example by spatial locations)
 * @param mZ (matrix): (T x s) matrix of smoothed state vectors
 * @param vbeta (vector) (p x 1) matrix (i.e. a vector) of fixed effects coef,
 * does NOT change with time
 * @param mXz (array) (s x s) non scaled transfer matrix (assumed constant in time
 * (the complete transfer matrix is scaled by alpha)
 * @param cPsm (array): (s x s x T) array of smoothed state variance matrices,
 *  each of those is accessed by cPsm.slice(t)
 */

double AlphaUpdate(const arma::mat & mY_fixed_res,
                   const arma::mat & mZ,
                   const arma::mat & mXz,
                   const arma::cube & cPsm){

  int T = mY_fixed_res.n_cols;

  double num = 0.0;
  double den = 0.0;

  for(int t = 0; t < T; t++){
    // NOTE: (mXz * mZ.col(t)) can be computed once and used also
    // in other updates
    num += arma::trace(mY_fixed_res.col(t) * (mXz * mZ.col(t)).t());
    den += arma::trace(mXz *
      (mZ.col(t) * mZ.col(t).t() + cPsm.slice(t)) * mXz.t());
  };

  // TO DO: add error message if den == 0
  return num / den;

}

/**
* @description compute the matrix S00
* @param smoothed_states (matrix): matrix of smoothed states
* @param smoothed_vars (array): array of smoothed sates variance matrices
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
* of the nondiffuse part of the initial state vector.
* @return (matrix) m x m
*/

arma::mat ComputeS00(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::vec & z0,
                     const arma::mat & P0){

  int T = z0.n_cols;

  arma::mat S00 = z0 * z0.t() + P0;

  // all except the last time T
  for(int t = 0; t < (T - 1); t++){
    S00 += smoothed_states.t() * smoothed_states.col(t).t() + smoothed_vars.slice(t);
  }

  return(S00);
}

/**
* @description compute the matrix S11
* @param smoothed_states (matrix): matrix of smoothed states
* @param smoothed_vars (array): array of smoothed sates variance matrices
* @param S00 (matrix) m x m as defined in the paper
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @param P0 (matrix) P0: starting value m x m matrix containing the covariance matrix
* of the nondiffuse part of the initial state vector.
* @return (matrix) m x m
*/

arma::mat ComputeS11(const arma::mat & smoothed_states,
                     const arma::cube & smoothed_vars,
                     const arma::mat & S00,
                     const arma::vec & z0,
                     const arma::mat & P0){
  int T = z0.n_cols;

  return(S00 - z0 * z0.t() - P0 +
         smoothed_states.col(T-1) * smoothed_states.col(T-1).t() + smoothed_vars.slice(T-1));
}

/**
* @description compute the matrix S11
* @param smoothed_states (matrix): matrix of smoothed states
* @param lagone_smoothed_covars (array): array of lag one
* smoothed sates covariance matrices
* @param z0 (matrix) z0: starting value m x 1 matrix (i.e. vector) containing the expected values of the initial states
* @return (matrix) m x m
*/

arma::mat ComputeS10(const arma::mat & smoothed_states,
                     const arma::cube & lagone_smoothed_covars,
                     const arma::vec & z0){

  int T = z0.n_cols;

  arma::mat S10 = smoothed_states.col(0) * z0.t() + lagone_smoothed_covars.slice(0);

  // all except the last time T
  for(int t = 1; t < T; t++){
    S10 += smoothed_states.col(t) * smoothed_states.col(t-1).t() + lagone_smoothed_covars.slice(t);
  }

  return(S10);
}


// eq. (7)
// where S10 and S11 are defined in the article
// and are function of the (Kalman) smoothed vectors
/**
* @description EM update for the g constant, this is g_HDGM in the HDGM paper
* i.e. the autoregressive coefficient
* @param S00 (matrix): given the kalman smoothed states
* matrix_sum(for(t in 1:T)({z_{t-1} %*% t(z_{t-1}) + P_{t-1}}) and
* P_{t-1} is the smoothed variance at time t-1
* @param S10 (matrix): given the kalman smoothed states
* matrix_sum(for(t in 1:T)({z_{t} %*% t(z_{t-1}) + P_{t,t-1}}) and
* P_{t,t-1} is the smoothed covariance between times t and t-1
*
* @return (num)
*/

double gUpdate(const arma::mat & S00,
               const arma::mat & S10){
  return(arma::sum(arma::trace(S10)) / arma::sum(arma::trace(S00)));
}

// NOTE: without missing data Omega_one_t = Omega_t

// NOTE: as in alpha Update, for each iteration
// mXzt * zZt can be computed once

/**
* @description Omega_one_t definition (A.3)
*
* all vector and matrices are taken only in the non missing rows
* @param yt (matrix): (q x 1) matrix (i.e. vector) of observed vector at time t, NA allowed
* (a sorting is assumed, example by spatial locations)
* @param zt (matrix): (s x 1) matrix (i.e. vector) of smoothed state vector at time t
* @param alpha (num): scaling factor of the state vector on the observed vector
* in the HDGM model this is the upsilon parameter
* @param Xzt (matrix) (s x s) unscaled transfer matrix at time t
* @param Pt (matrix): (s x s) matrix of smoothed state variance at time t
*/

arma::mat Omega_one_t(const arma::vec & vY_fixed_res_t,
                      const arma::vec & vZt,
                      const arma::mat & mXz,
                      const arma::mat & mPsmt,
                      double alpha){

  arma::vec res = vY_fixed_res_t - alpha * mXz * vZt;

    return (res * res.t() +
            alpha * alpha * mXz * mPsmt * mXz.t());

}

// TO DO: Omega_t function in case there are some missing observations
// Along with Permutation matrix D definition

// here assuming NOT missing values
// NOTE: maybe it's also possible to define it just as a matrix
// considering only the elements in the diagonal
// since then the trace is taken
arma::mat OmegaSumUpdate(const arma::mat & mY_fixed_res,
                        const arma::mat & Zt,
                        const arma::mat & mXz,
                        const arma::cube & cPsmt,
                        double alpha){

  int T = mY_fixed_res.n_cols;
  int n = mY_fixed_res.n_rows;


  arma::mat Omega_sum(n, n, arma::fill::zeros);

  for(int t = 0; t < T; t++){
    Omega_sum += Omega_one_t(mY_fixed_res.col(t),
                             Zt.col(t),
                             mXz,
                             cPsmt.slice(t),
                             alpha);
  };

  return(Omega_sum);


};

double Sigma2Update(const arma::mat& Omega_sum,
                    const int n, // dimension of observation vector
                    const int T){ // number of observations

  // TO DO: take sum of traces instead of trace of sums
  return (arma::trace(Omega_sum) / (T * n));
};

/**
 * @brief Computes the negative expected complete-data log-likelihood
 *        (up to a constant) for the HDGM model, to be minimized over theta.
 *
 * @param theta Spatial decay parameter to evaluate
 * @param dist_matrix Distance matrix between spatial locations (p x p)
 * @param S00 Smoothed second moment of z_{t-1} (p x p)
 * @param S10 Smoothed cross-moment between z_t and z_{t-1} (p x p)
 * @param S11 Smoothed second moment of z_t (p x p)
 * @param g Autoregressive coefficient (scalar)
 * @param N Number of time observations (T)
 * @return double The value of the negative objective function at given theta
 */

double negative_to_optim(double theta,
                         const arma::mat& dist_matrix,
                         const arma::mat& S00,
                         const arma::mat& S10,
                         const arma::mat& S11,
                         double g,
                         int N) {
  arma::mat Sigma_eta = ExpCor(dist_matrix, theta);

  double logdet_val = 0.0;
  double sign = 0.0;


  // NOTE: for small Sigma_eta one can keep the function like this
  // for big Sigma_eta it's convenient to compute a (ex. Cholesky) decomposition
  // of Sigma_eta once and use it to compute both the log determinant and the inverse
  arma::log_det(logdet_val, sign, Sigma_eta);

  arma::mat Sigma_eta_inv = arma::inv_sympd(Sigma_eta);
  arma::mat expr = S11 - g * S10 - g * S10.t() + g * g * S00;

  double trace_val = arma::trace(Sigma_eta_inv * expr);

  return N * logdet_val + trace_val;
}

/**
 * @brief Minimize a scalar function using Brent's method.
 *
 * @param f         Function to minimize. Must be a callable of type double -> double.
 * @param ax        Lower bound of the search interval.
 * @param bx        Initial guess (should lie between ax and cx).
 * @param cx        Upper bound of the search interval.
 * @param tol       Desired tolerance for convergence (default: 1e-8).
 * @param max_iter  Maximum number of iterations to perform (default: 100).
 * @return          The x-coordinate of the minimum.
 */
double local_min_rc ( double &a, double &b, int &status, double value )

//****************************************************************************80
//
//  Purpose:
//
//    LOCAL_MIN_RC seeks a minimizer of a scalar function of a scalar variable.
//
//  Discussion:
//
//    This routine seeks an approximation to the point where a function
//    F attains a minimum on the interval (A,B).
//
//    The method used is a combination of golden section search and
//    successive parabolic interpolation.  Convergence is never much
//    slower than that for a Fibonacci search.  If F has a continuous
//    second derivative which is positive at the minimum (which is not
//    at A or B), then convergence is superlinear, and usually of the
//    order of about 1.324...
//
//    The routine is a revised version of the Brent local minimization
//    algorithm, using reverse communication.
//
//    It is worth stating explicitly that this routine will NOT be
//    able to detect a minimizer that occurs at either initial endpoint
//    A or B.  If this is a concern to the user, then the user must
//    either ensure that the initial interval is larger, or to check
//    the function value at the returned minimizer against the values
//    at either endpoint.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 July 2011
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters
//
//    Input/output, double &A, &B.  On input, the left and right
//    endpoints of the initial interval.  On output, the lower and upper
//    bounds for an interval containing the minimizer.  It is required
//    that A < B.
//
//    Input/output, int &STATUS, used to communicate between
//    the user and the routine.  The user only sets STATUS to zero on the first
//    call, to indicate that this is a startup call.  The routine returns STATUS
//    positive to request that the function be evaluated at ARG, or returns
//    STATUS as 0, to indicate that the iteration is complete and that
//    ARG is the estimated minimizer.
//
//    Input, double VALUE, the function value at ARG, as requested
//    by the routine on the previous call.
//
//    Output, double LOCAL_MIN_RC, the currently considered point.
//    On return with STATUS positive, the user is requested to evaluate the
//    function at this point, and return the value in VALUE.  On return with
//    STATUS zero, this is the routine's estimate for the function minimizer.
//
//  Local parameters:
//
//    C is the squared inverse of the golden ratio.
//
//    EPS is the square root of the relative machine precision.
//
{
  static double arg;
  static double c;
  static double d;
  static double e;
  static double eps;
  static double fu;
  static double fv;
  static double fw;
  static double fx;
  static double midpoint;
  static double p;
  static double q;
  static double r;
  static double tol;
  static double tol1;
  static double tol2;
  static double u;
  static double v;
  static double w;
  static double x;
//
//  STATUS (INPUT) = 0, startup.
//
  if ( status == 0 )
  {
    if ( b <= a )
    {
      std::cout << "\n";
       std::cout << "LOCAL_MIN_RC - Fatal error!\n";
       std::cout << "  A < B is required, but\n";
       std::cout << "  A = " << a << "\n";
       std::cout << "  B = " << b << "\n";
      status = -1;
      exit ( 1 );
    }
    c = 0.5 * ( 3.0 - sqrt ( 5.0 ) );

    eps = sqrt (r8_epsilon ( ) );
    tol = r8_epsilon ( );

    v = a + c * ( b - a );
    w = v;
    x = v;
    e = 0.0;

    status = 1;
    arg = x;

    return arg;
  }
//
//  STATUS (INPUT) = 1, return with initial function value of FX.
//
  else if ( status == 1 )
  {
    fx = value;
    fv = fx;
    fw = fx;
  }
//
//  STATUS (INPUT) = 2 or more, update the data.
//
  else if ( 2 <= status )
  {
    fu = value;

    if ( fu <= fx )
    {
      if ( x <= u )
      {
        a = x;
      }
      else
      {
        b = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    }
    else
    {
      if ( u < x )
      {
        a = u;
      }
      else
      {
        b = u;
      }

      if ( fu <= fw || w == x )
      {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      }
      else if ( fu <= fv || v == x || v == w )
      {
        v = u;
        fv = fu;
      }
    }
  }
//
//  Take the next step.
//
  midpoint = 0.5 * ( a + b );
  tol1 = eps * fabs ( x ) + tol / 3.0;
  tol2 = 2.0 * tol1;
//
//  If the stopping criterion is satisfied, we can exit.
//
  if ( fabs ( x - midpoint ) <= ( tol2 - 0.5 * ( b - a ) ) )
  {
    status = 0;
    return arg;
  }
//
//  Is golden-section necessary?
//
  if ( fabs ( e ) <= tol1 )
  {
    if ( midpoint <= x )
    {
      e = a - x;
    }
    else
    {
      e = b - x;
    }
    d = c * e;
  }
//
//  Consider fitting a parabola.
//
  else
  {
    r = ( x - w ) * ( fx - fv );
    q = ( x - v ) * ( fx - fw );
    p = ( x - v ) * q - ( x - w ) * r;
    q = 2.0 * ( q - r );
    if ( 0.0 < q )
    {
      p = - p;
    }
    q = fabs ( q );
    r = e;
    e = d;
//
//  Choose a golden-section step if the parabola is not advised.
//
    if (
      ( fabs ( 0.5 * q * r ) <= fabs ( p ) ) ||
      ( p <= q * ( a - x ) ) ||
      ( q * ( b - x ) <= p ) )
    {
      if ( midpoint <= x )
      {
        e = a - x;
      }
      else
      {
        e = b - x;
      }
      d = c * e;
    }
//
//  Choose a parabolic interpolation step.
//
    else
    {
      d = p / q;
      u = x + d;

      if ( ( u - a ) < tol2 )
      {
        d = tol1 * r8_sign ( midpoint - x );
      }

      if ( ( b - u ) < tol2 )
      {
        d = tol1 * r8_sign ( midpoint - x );
      }
    }
  }
//
//  F must not be evaluated too close to X.
//
  if ( tol1 <= fabs ( d ) )
  {
    u = x + d;
  }
  if ( fabs ( d ) < tol1 )
  {
    u = x + tol1 * r8_sign ( d );
  }
//
//  Request value of F(U).
//
  arg = u;
  status = status + 1;

  return arg;

  
}/**
 * @brief Performs the EM update of the spatial decay parameter theta
 *        in the Hierarchical Dynamic Gaussian Model (HDGM).
 *
 * @param dist_matrix p x p distance matrix between spatial locations
 * @param g Autoregressive coefficient of the hidden state process
 * @param S00 Smoothed second moment of z_{t-1} over time (p x p)
 * @param S10 Smoothed cross-moment of z_t and z_{t-1} over time (p x p)
 * @param S11 Smoothed second moment of z_t over time (p x p)
 * @param theta0 Initial guess for theta (not used in optimization directly)
 * @param N Number of time points
 * @param lower lower bound for theta optimization
 * @param upper Upper bound for theta optimization
 * @return double Optimized value of theta that minimizes the objective
 */
double ThetaUpdate(const arma::mat& dist_matrix,
                   double g,
                   const arma::mat& S00,
                   const arma::mat& S10,
                   const arma::mat& S11,
                   double theta0,
                   int N,
                   double lower,
                   double upper) {

  auto obj_fun = [&](double theta) {
    return negative_to_optim(theta, dist_matrix, S00, S10, S11, g, N);
  };

  double result = brent_minimize(obj_fun, lower, theta0, upper); // avoid theta = 0
  return result;
}

/**
 * @brief EM update for fixed-effect coefficients (Equation 5 in HDGM model), no missing values version.
 *
 * Estimates the fixed effects `beta` given full observation matrix `y`, smoothed state vectors `z`,
 * and covariate matrices `Xbeta`, assuming no missing data.
 *
 * @param Xbeta Array of shape (T elements of q x p), covariate matrices for each time point
 * @param y Matrix of observed vectors (q x T), no missing values assumed
 * @param z Matrix of smoothed hidden states (s x T)
 * @param alpha Scalar coefficient (upsilon parameter in HDGM)
 * @param Xz Transfer matrix from state to observation space (q x s)
 * @param inv_mXbeta_sum Precomputed inverse of ∑ₜ Xbetaᵗ Xbeta (p x p)
 *
 * @return arma::vec Estimated fixed-effect coefficients (p x 1)
 */
arma::vec BetaUpdate(const arma::cube& Xbeta,  // T elements of (q x p)
                     const arma::mat& y,                    // (q x T)
                     const arma::mat& z,                    // (s x T)
                     double alpha,
                     const arma::mat& Xz,                   // (q x s)
                     const arma::mat& inv_mXbeta_sum)       // (p x p)
{
  int p = inv_mXbeta_sum.n_rows;
  arma::vec right_term = arma::zeros(p);

  int T = y.n_cols;

  for (int t = 0; t < T; ++t) {
    right_term += Xbeta.slice(t).t() * (y.col(t) - alpha * Xz * z.col(t)); // (p)
  }

  return inv_mXbeta_sum * right_term;
}
