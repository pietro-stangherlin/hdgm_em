#include "src/optim/nelder_mead.h"

double function_to_minimize(const std::array<double,2>& x) {
  return x[0]*x[0] + x[1]*x[1];
}

int main() {

  std::array<double,2> start = { 0.5, 0.5 };
  std::array<double,2> step = { 0.1, 0.1 };

  nelder_mead_result<double,2> result = nelder_mead<double,2>(
    function_to_minimize,
    start,
    1.0e-5, // the terminating limit for the variance of function values
    step
  );
  std::cout << "Found minimum: " << std::fixed << result.xmin[0] << ' ' << result.xmin[1] << std::endl;
  std::cout<< "Expected (0,0)"

  return 0;
}

