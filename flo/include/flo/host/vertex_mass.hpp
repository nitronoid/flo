#ifndef FLO_HOST_INCLUDED_VERTEX_MASS
#define FLO_HOST_INCLUDED_VERTEX_MASS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <igl/doublearea.h>

FLO_HOST_NAMESPACE_BEGIN

/// @breif Computes the vertex masses
//  @param V #Vx3 A column major matrix of vertex positions
//  @param F #Fx3 A column major matrix of face vertex indices
//  @param M #V An array of vertex masses
template <typename DerivedV, typename DerivedF, typename DerivedM>
FLO_API void vertex_mass(const Eigen::MatrixBase<DerivedV>& V,
                         const Eigen::MatrixBase<DerivedF>& F,
                         Eigen::PlainObjectBase<DerivedM>& M);

#include "vertex_mass.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_VERTEX_MASS

