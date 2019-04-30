#ifndef FLO_HOST_INCLUDED_DIVERGENT_EDGES
#define FLO_HOST_INCLUDED_DIVERGENT_EDGES

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_quaternion_operation.hpp"

FLO_HOST_NAMESPACE_BEGIN

/// @breif Computes new edges that best fit the provided similarity transforms.
//  @param V #Vx3 A column major matrix of vertex positions
//  @param F #Fx3 A column major matrix of face vertex indices
//  @param X #Xx4 A column major matrix of quaternion transforms
//  @param L #Vx#V Sparse cotangent laplacian matrix
//  @param E #Ex3 A column major matrix of best fit edges
template <typename DerivedV,
          typename DerivedF,
          typename DerivedX,
          typename DerivedE>
FLO_API void divergent_edges(const Eigen::MatrixBase<DerivedV>& V,
                             const Eigen::MatrixBase<DerivedF>& F,
                             const Eigen::MatrixBase<DerivedX>& X,
                             const Eigen::SparseMatrix<real>& L,
                             Eigen::PlainObjectBase<DerivedE>& E);

#include "divergent_edges.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_DIVERGENT_EDGES

