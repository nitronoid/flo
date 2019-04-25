#ifndef FLO_INCLUDED_WILLMORE_FLOW
#define FLO_INCLUDED_WILLMORE_FLOW

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include "flo/host/willmore_flow.hpp"
#include "flo/host/mean_curvature.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/spin_xform.hpp"
#include "flo/host/vertex_mass.hpp"
#include "flo/host/orthonormalize.hpp"
#include "flo/host/project_basis.hpp"
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV, typename DerivedF, typename BinaryOp>
FLO_API void willmore_flow(Eigen::MatrixBase<DerivedV>& V,
                           const Eigen::MatrixBase<DerivedF>& F,
                           BinaryOp integrator);

#include "willmore_flow.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_INCLUDED_WILLMORE_FLOW

