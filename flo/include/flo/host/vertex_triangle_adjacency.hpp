#ifndef FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/vertex_triangle_adjacency.h>
//#include <numeric>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedF,
          typename DerivedVTAK,
          typename DerivedVTA,
          typename DerivedVTV,
          typename DerivedVTCV>
FLO_API void
vertex_triangle_adjacency(const Eigen::MatrixBase<DerivedF>& F,
                          Eigen::PlainObjectBase<DerivedVTAK>& VTAK,
                          Eigen::PlainObjectBase<DerivedVTA>& VTA,
                          Eigen::PlainObjectBase<DerivedVTV>& VTV,
                          Eigen::PlainObjectBase<DerivedVTCV>& VTCV);

#include "vertex_triangle_adjacency.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

