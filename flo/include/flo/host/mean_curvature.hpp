#ifndef FLO_HOST_INCLUDED_MEAN_CURVATURE
#define FLO_HOST_INCLUDED_MEAN_CURVATURE

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <algorithm>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV, typename DerivedM, typename DerivedHN>
FLO_API void mean_curvature_normal(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   Eigen::PlainObjectBase<DerivedHN>& HN);

template <typename DerivedV, typename DerivedM, typename DerivedH>
FLO_API void mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                            const Eigen::SparseMatrix<real>& L,
                            const Eigen::MatrixBase<DerivedM>& M,
                            Eigen::PlainObjectBase<DerivedH>& H);

template <typename DerivedV,
          typename DerivedM,
          typename DerivedN,
          typename DerivedH>
FLO_API void signed_mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   const Eigen::MatrixBase<DerivedN>& N,
                                   Eigen::PlainObjectBase<DerivedH>& H);

#include "mean_curvature.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_MEAN_CURVATURE

