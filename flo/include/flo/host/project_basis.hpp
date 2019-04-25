#ifndef FLO_HOST_INCLUDED_PROJECT_BASIS
#define FLO_HOST_INCLUDED_PROJECT_BASIS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV, typename DerivedU, typename BinaryOp>
FLO_API void project_basis(Eigen::MatrixBase<DerivedV>& V,
                           const Eigen::MatrixBase<DerivedU>& U,
                           BinaryOp inner_product);

#include "project_basis.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_PROJECT_BASIS
