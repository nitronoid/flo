#ifndef FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/vertex_triangle_adjacency.h>
//#include <numeric>

FLO_HOST_NAMESPACE_BEGIN

/// @breif Computes the vertex triangle adjacency with keys, the valence and the
// cumulative valence per vertex
//  @param F #Fx3 A column major matrix of face vertex indices
//  @param VTAK An array containing the vertex triangle adjacency keys
//  @param VTA  An array containing the vertex triangle adjacency 
//  @param VTV  #V An array containing the vertex triangle valence
//  @param VTCV #V An array containing the vertex triangle cumulative valence
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

