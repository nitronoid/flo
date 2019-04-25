#ifndef FLO_HOST_INCLUDED_VERTEX_VERTEX_ADJACENCY
#define FLO_HOST_INCLUDED_VERTEX_VERTEX_ADJACENCY

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <Eigen/Sparse>
#include <igl/adjacency_matrix.h>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedF,
          typename DerivedVVAK,
          typename DerivedVVA,
          typename DerivedVVV,
          typename DerivedVVCV>
FLO_API void vertex_vertex_adjacency(const Eigen::MatrixBase<DerivedF>& F,
                                     Eigen::PlainObjectBase<DerivedVVAK>& VVAK,
                                     Eigen::PlainObjectBase<DerivedVVA>& VVA,
                                     Eigen::PlainObjectBase<DerivedVVV>& VVV,
                                     Eigen::PlainObjectBase<DerivedVVCV>& VVCV);

#include "vertex_vertex_adjacency.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_VERTEX_VERTEX_ADJACENCY
