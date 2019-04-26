#ifndef FLO_HOST_INCLUDED_SPIN_XFORM
#define FLO_HOST_INCLUDED_SPIN_XFORM

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/spin_xform.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/vertex_vertex_adjacency.hpp"
#include "flo/host/similarity_xform.hpp"
#include "flo/host/divergent_edges.hpp"
#include "flo/host/spin_positions.hpp"
#include <igl/doublearea.h>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV, typename DerivedF, typename DerivedP>
FLO_API void spin_xform(Eigen::MatrixBase<DerivedV>& V,
                        const Eigen::MatrixBase<DerivedF>& F,
                        const Eigen::MatrixBase<DerivedP>& P,
                        const Eigen::SparseMatrix<real>& L);

#include "spin_xform.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SPIN_XFORM
