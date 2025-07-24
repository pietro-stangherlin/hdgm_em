#pragma once

#include "Kalman_types.h"

template <typename CovStore>
KalmanFilterResultT<CovStore> SKF_core(const KalmanFilterInput& kf_inp, CovStore& Pp_store, CovStore& Pf_store);

template <typename CovStore>
KalmanSmootherResultT<CovStore> FIS_core(const KalmanSmootherInputT<CovStore>& ksm_inp, CovStore& Ps, CovStore& Plos);
