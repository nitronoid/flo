#ifndef FLO_HOST_INCLUDED_DIVERGENT_EDGES
#define FLO_HOST_INCLUDED_DIVERGENT_EDGES

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_quaternion_operation.hpp"

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV,
          typename DerivedF,
          typename DerivedH,
          typename DerivedE>
FLO_API void divergent_edges(const Eigen::MatrixBase<DerivedV>& V,
                             const Eigen::MatrixBase<DerivedF>& F,
                             const Eigen::MatrixBase<DerivedH>& h,
                             const Eigen::SparseMatrix<real>& L,
                             Eigen::PlainObjectBase<DerivedE>& E);

#include "divergent_edges.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_DIVERGENT_EDGES

