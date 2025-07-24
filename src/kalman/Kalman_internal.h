#pragma once

#include "Kalman_types.h"
#include "Kalman_internal.hpp"

KalmanFilterResult SKF_cpp(const KalmanFilterInput& kf_inp);

KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp);

KalmanSmootherLlikResult SKFS_cpp(const KalmanFilterInput& kfsm_inp);

KalmanFilterResultMat SKF_cpp_mat(const KalmanFilterInput& kf_inp);

KalmanSmootherResultMat FIS_cpp_mat(const KalmanSmootherInputMat& ksm_inp);

KalmanSmootherLlikResultMat SKFS_cpp_mat(const KalmanFilterInput& kfsm_inp);
