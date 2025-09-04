#pragma once

#include "EM_types.hpp"

#include <RcppArmadillo.h>
#include <stdio.h>
#include <limits>

#include "../kalman/Kalman_internal.hpp"
#include "EM_types.hpp"
#include "EM_functions.hpp"
#include "EM_algorithm_impl.hpp"
#include "../utils/covariances.h"
#include "../utils/symmetric_matr_vec.h"

template <typename CovStore>
EMOutputUnstructured UnstructuredEM_cpp_core(EMInputUnstructured& em_in);

template <typename CovStore>
EMOutput EMHDGM_cpp_core(EMInput& em_in);

EMOutputUnstructured UnstructuredEM_cpp(EMInputUnstructured& em_in);

EMOutputUnstructured UnstructuredEM_cpp_mat(EMInputUnstructured& em_in);

EMOutput EMHDGM_cpp(EMInput & em_in);
EMOutput EMHDGM_cpp_mat(EMInput & em_in);

EMOutput EMHDGM_state_scale_cpp(EMInput_statescale & em_in);
EMOutput EMHDGM_state_scale_cpp_mat(EMInput_statescale & em_in);

EMOutput EMHDGM_diag_cpp(EMInput & em_in);
EMOutput EMHDGM_diag_cpp_mat(EMInput & em_in);

EMOutput EMHDGM_tv_cpp(EMInputNonConstCovariates & em_in);
EMOutput EMHDGM_tv_cpp_mat(EMInputNonConstCovariates & em_in);
