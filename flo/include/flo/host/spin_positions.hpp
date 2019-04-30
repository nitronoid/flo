#ifndef FLO_HOST_INCLUDED_SPIN_POSITIONS
#define FLO_HOST_INCLUDED_SPIN_POSITIONS

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include "flo/host/flo_matrix_operation.hpp"

FLO_HOST_NAMESPACE_BEGIN

/// @breif Solves the final best fit positions, given a set of divergent edges
/// @param QL the real quaternion laplacian matrix
/// @param QE the divergent edges
/// @param V the output positions
template <typename DerivedQE, typename DerivedV>
FLO_API void spin_positions(const Eigen::SparseMatrix<real>& QL,
                            const Eigen::MatrixBase<DerivedQE>& QE,
                            Eigen::PlainObjectBase<DerivedV>& V);

#include "spin_positions.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SPIN_POSITIONS

