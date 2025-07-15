#include <RcppArmadillo.h>
#include <iostream>
#include <array>

#include "../../em/EM_functions.cpp"
#include "../../utils/covariances.cpp"


int main(){
  //// NOTE: maybe scenarions have not realistic values

  //////////// Scenario I ////////////////
  std::cout<< "Scenario I" << std::endl;
  std::cout<< "dim(state vector)  = 2" << std::endl;
  std::cout<< "Scenario S00 = S10 = S11 = I" << std::endl;

  // other parameters
  double g = 0.6;
  int T = 100;
  arma::mat dist_matrix(1, 1, arma::fill::eye);
  arma::mat S00(1, 1, arma::fill::eye);
  arma::mat S10(1, 1, arma::fill::eye);
  arma::mat S11(1, 1, arma::fill::eye);

  std::array<double,2> start = {5.0, 20.0 };
  std::array<double,2> step = { 0.1, 0.1 };
  double var_terminating_lim = 1e-10;

  std::array<double,2> theta_v_temp = ThetaVUpdate(dist_matrix,
                              g,
                              T,
                              S00, S10, S11,
                              start,
                              step,
                              var_terminating_lim);

  std::cout << "ThetaVUpdate result: " << theta_v_temp[0] << std::endl;
  std::cout << "ThetaVUpdate result: " << theta_v_temp[1] << std::endl;

  return 0;

}
