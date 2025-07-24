#pragma once

#include <RcppArmadillo.h>
#include <stdio.h>
#include <limits>

#include "../kalman/Kalman_internal.h"
#include "EM_algorithm.h"
#include "EM_functions.h"
#include "EM_functions.hpp"
#include "../utils/covariances.h"
#include "../utils/symmetric_matr_vec.h"

template <typename CovStore>
EMOutputUnstructured UnstructuredEM_cpp_core(EMInputUnstructured& em_in);


