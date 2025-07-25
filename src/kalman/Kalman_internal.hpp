#pragma once

#include "Kalman_types.hpp"

template <typename CovStore>
KalmanFilterResultT<CovStore> SKF_core(const KalmanFilterInput& kf_inp, CovStore& Pp_store, CovStore& Pf_store);

template <typename CovStore>
KalmanSmootherResultT<CovStore> FIS_core(const KalmanSmootherInputT<CovStore>& ksm_inp, CovStore& Ps, CovStore& Plos);

KalmanFilterResult SKF_cpp(const KalmanFilterInput& kf_inp);

KalmanSmootherResult FIS_cpp(const KalmanSmootherInput& ksm_inp);



KalmanFilterResultMat SKF_cpp_mat(const KalmanFilterInput& kf_inp);

KalmanSmootherResultMat FIS_cpp_mat(const KalmanSmootherInputMat& ksm_inp);

// function overloading: only return type changes

KalmanSmootherLlikResult SKFS_cpp(const KalmanFilterInput& kfsm_inp,
                                  std::type_identity<arma::cube>);

KalmanSmootherLlikResultMat SKFS_cpp(const KalmanFilterInput& kfsm_inp,
                                     std::type_identity<arma::mat>);
