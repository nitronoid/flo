#ifndef FLO_HOST_INCLUDED_ORTHONORMALIZE
#define FLO_HOST_INCLUDED_ORTHONORMALIZE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV, typename BinaryOp, typename DerivedU>
FLO_API void orthonormalize(const Eigen::MatrixBase<DerivedV>& V,
                            BinaryOp inner_product,
                            Eigen::PlainObjectBase<DerivedU>& U);

#include "orthonormalize.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_ORTHONORMALIZE

