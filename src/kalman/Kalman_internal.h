#pragma once

#include "Kalman_types.h"

KalmanFilterResult SKF_cpp(const KalmanFilterInput& kf_inp);

KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp);

KalmanSmootherResult SKFS_cpp(const KalmanFilterInput& kfsm_inp);
