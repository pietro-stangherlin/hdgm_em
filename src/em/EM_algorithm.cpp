#include <RcppArmadillo.h>
#include <stdio.h>
#include <limits>

#include "../kalman/Kalman_internal.hpp"
#include "EM_functions.hpp"
#include "EM_algorithm.hpp"
#include "EM_functions_impl.hpp"
#include "../utils/covariances.h"

// -------------------- Unstructured Case --------------------- //
// from Parameter estimation for linear dynamical systems.
// Ghahramani, Z. & Hinton, G. E. Technical Report Technical Report CRG-TR-96-2,
// University of Totronto, Dept. of Computer Science, 1996.

EMOutputUnstructured UnstructuredEM_cpp(EMInputUnstructured& em_in){
  return UnstructuredEM_cpp_core<arma::cube>(em_in);
};


EMOutputUnstructured UnstructuredEM_cpp_mat(EMInputUnstructured& em_in){
  return UnstructuredEM_cpp_core<arma::mat>(em_in);
};


//  ------------------ Structured Case ------------------------ //
EMOutput EMHDGM_cpp(EMInput & em_in){
  return EMHDGM_cpp_core<arma::cube>(em_in);
};

EMOutput EMHDGM_cpp_mat(EMInput & em_in){
  return EMHDGM_cpp_core<arma::mat>(em_in);
};

//  ------------------ Structured Case diag transition matrix ------------------------ //

EMOutput EMHDGM_diag_cpp(EMInput & em_in){
  return EMHDGM_diag_cpp_core<arma::cube>(em_in);
};

EMOutput EMHDGM_diag_cpp_mat(EMInput & em_in){
  return EMHDGM_diag_cpp_core<arma::mat>(em_in);
};

// --------------- Structured AR + Random Effects Covariates Coefficients --------------//
EMOutput EMHDGM_tv_cpp(EMInputNonConstCovariates & em_in){
  return EMHDGM_AR_RandomEffects_cpp_core<arma::cube>(em_in);
};

EMOutput EMHDGM_tv_cpp_mat(EMInputNonConstCovariates & em_in){
  return EMHDGM_AR_RandomEffects_cpp_core<arma::mat>(em_in);
};

