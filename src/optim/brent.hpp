#pragma once
#include <functional>

namespace brent {

    
double local_min_rc ( double &a, double &b, int &status, double value );
double r8_epsilon ( );
double r8_sign ( double x );

double brent_minimize(const std::function<double(double)>& f,
 double a_init, double b_init,
  int max_iterations = 100);

}

