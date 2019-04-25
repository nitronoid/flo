#ifndef FLO_HOST_INCLUDED_SPIN_POSITIONS
#define FLO_HOST_INCLUDED_SPIN_POSITIONS

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <Eigen/CholmodSupport>

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedQE, typename DerivedV>
FLO_API void spin_positions(const Eigen::SparseMatrix<real>& QL,
                            const Eigen::MatrixBase<DerivedQE>& QE,
                            Eigen::PlainObjectBase<DerivedV>& V);

#include "spin_positions.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SPIN_POSITIONS

