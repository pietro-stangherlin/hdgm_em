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
EMOutput EMHDGM_cpp_core(EMInput& em_in)

// change return type
EMOutput EMHDGM_cpp(EMInput em_in);

EMOutputUnstructured UnstructuredEM_cpp(EMInputUnstructured& em_in);

EMOutputUnstructured UnstructuredEM_cpp_mat(EMInputUnstructured& em_in);
