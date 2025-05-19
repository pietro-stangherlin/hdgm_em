#pragma once

#include"Kalman_types.h"

KalmanFilterResult SKF_cpp(KalmanFilterInput inp);

KalmanSmootherResult FIS_cpp(KalmanSmootherInput inp);

KalmanSmootherResult SKFS_cpp(KalmanFilterInput inp);
