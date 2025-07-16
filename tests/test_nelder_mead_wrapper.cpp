// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <iostream>
#include "../src/optim/nelder_mead.h"

// minimum is (0,0)
double function_to_minimize(const std::array<double,2>& x) {
  return x[0]*x[0] + x[1]*x[1];
}

// minimum is (0,0)
double function_to_minimize_additional_args(const std::array<double,2>& x,
                                            double true_min) {
  return  x[0]*x[0] + x[1]*x[1] - true_min;
}

// [[Rcpp::export]]
Rcpp::List MyNelderMead(double x1_start,
                        double x2_start,
                        double x1_step,
                        double x2_step){

  std::array<double,2> start = { x1_start, x2_start};
  std::array<double,2> step = { x2_step, x2_step };

  nelder_mead_result<double,2> result = nelder_mead<double,2>(
    function_to_minimize,
    start,
    1.0e-10, // the terminating limit for the variance of function values
    step
  );

  return Rcpp::List::create(
    Rcpp::Named("x1_min") = result.xmin[1],
    Rcpp::Named("x2_min") = result.xmin[2]
  );

}


// [[Rcpp::export]]
Rcpp::List MyNelderMeadAdditionalArgs(double x1_start,
                        double x2_start,
                        double x1_step,
                        double x2_step,
                        double true_min){

  std::array<double,2> start = { x1_start, x2_start};
  std::array<double,2> step = { x2_step, x2_step };

  auto obj_fun = [&](const std::array<double,2>& x) {
    return function_to_minimize_additional_args(x, true_min);
  };

  nelder_mead_result<double,2> result = nelder_mead<double,2>(
    obj_fun,
    start,
    1.0e-10, // the terminating limit for the variance of function values
    step
  );

  return Rcpp::List::create(
    Rcpp::Named("x1_min") = result.xmin[1],
    Rcpp::Named("x2_min") = result.xmin[2]
  );

}

