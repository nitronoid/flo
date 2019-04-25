#ifndef FLO_HOST_INCLUDED_INTRINSIC_DIRAC
#define FLO_HOST_INCLUDED_INTRINSIC_DIRAC

#include "flo/flo_internal.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/flo_quaternion_operation.hpp"

FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedV,
          typename DerivedF,
          typename DerivedVV,
          typename DerivedA,
          typename DerivedP>
FLO_API void intrinsic_dirac(const Eigen::MatrixBase<DerivedV>& V,
                             const Eigen::MatrixBase<DerivedF>& F,
                             const Eigen::MatrixBase<DerivedVV>& VV,
                             const Eigen::MatrixBase<DerivedA>& A,
                             const Eigen::MatrixBase<DerivedP>& P,
                             Eigen::SparseMatrix<real>& D);

#include "intrinsic_dirac.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_INTRINSIC_DIRAC
