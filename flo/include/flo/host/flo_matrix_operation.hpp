#ifndef FLO_HOST_INCLUDED_MATRIX_OPERATION
#define FLO_HOST_INCLUDED_MATRIX_OPERATION

#include "flo/flo_internal.hpp"
#include "flo/host/flo_quaternion_operation.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

template <int R, int C>
FLO_API void insert_block_sparse(const Eigen::Matrix<real, R, C>& i_block,
                                 Eigen::SparseMatrix<real>& i_mat,
                                 int i_x,
                                 int i_y);

inline FLO_API Eigen::SparseMatrix<real>
to_real_quaternion_matrix(const Eigen::SparseMatrix<real>& i_real_matrix);

#include "flo_matrix_operation.cpp"  //template definitions

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_MATRIX_OPERATION
