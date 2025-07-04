# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iostream>
#include <functional>

#include "../optim/brent.hpp"

using namespace std;

// https://people.math.sc.edu/Burkardt/cpp_src/brent/brent.html


namespace brent {
double r8_epsilon ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_EPSILON returns the R8 roundoff unit.
//
//  Discussion:
//
//    The roundoff unit is a number R which is a power of 2 with the
//    property that, to the precision of the computer's arithmetic,
//      1 < 1 + R
//    but
//      1 = ( 1 + R / 2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    01 September 2012
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_EPSILON, the R8 round-off unit.
//
{
  const double value = 2.220446049250313E-016;

  return value;
}
//****************************************************************************80
double r8_sign ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_SIGN returns the sign of an R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the number whose sign is desired.
//
//    Output, double R8_SIGN, the sign of X.
//
{
  double value;

  if ( x < 0.0 )
  {
    value = -1.0;
  }
  else
  {
    value = 1.0;
  }
  return value;
}
//****************************************************************************80


/**
 * @brief Step to minimize a scalar function using Brent's method.
 * this function is used in a cycle
 *
 * @param f         Function to minimize. Must be a callable of type double -> double.
 * @param a        Lower bound of the search interval.
 * @param b        Upper bound of the search interval.
 * @param status
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

    eps = sqrt (brent::r8_epsilon ( ) );
    tol = brent::r8_epsilon ( );

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
        d = tol1 * brent::r8_sign ( midpoint - x );
      }

      if ( ( b - u ) < tol2 )
      {
        d = tol1 * brent::r8_sign ( midpoint - x );
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
    u = x + tol1 * brent::r8_sign ( d );
  }
//
//  Request value of F(U).
//
  arg = u;
  status = status + 1;

  return arg;
};

  /**
 * @brief minimize a scalar function using Brent's method.
 * this function is used in a cycle
 *
 * @param f         Function to minimize. Must be a callable of type double -> double.
 * @param a_init        Lower bound of the search interval.
 * @param b_init      Upper bound of the search interval.
 * @param max_iterations  maximum iteration number
 * @return          The x-coordinate of the minimum.
 */

double brent_minimize(const std::function<double(double)>& f,
                        double a_init, double b_init,
                        int max_iterations) {
    double a = a_init;
    double b = b_init;
    int status = 0;
    double value = 0.0; // CHECK starting value, maybe 0 is not appropriate
    double x;
    int iterations = 0;

    while (iterations <= max_iterations) {

        x = brent::local_min_rc(a, b, status, value);

        if (status == 0) {
            break; // Found the minimizer
        }

        value = f(x);
        iterations++;
    };

    if (iterations > max_iterations) {
            cerr << "Brent Warning: Maximum number of iterations reached.\n";
        };

    return x;
};

};
