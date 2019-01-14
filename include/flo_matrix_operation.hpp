#ifndef FLO_INCLUDED_MATRIX_OPERATION
#define FLO_INCLUDED_MATRIX_OPERATION

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

template <int R, int C>
void insert_block_sparse(
    const Eigen::Matrix<double, R, C>& i_block,
    Eigen::SparseMatrix<double>& i_mat,
    uint i_x,
    uint i_y);

template <int R, int C, int IR, int IC>
void insert_block_dense(
    const Eigen::Matrix<double, R, C>& i_block,
    Eigen::Matrix<double, IR, IC>& i_mat,
    uint i_x,
    uint i_y);

template <typename T, int R, int C>
std::vector<Eigen::Matrix<T, C, 1>> matrix_to_array(
    const Eigen::Matrix<T, R, C>& i_mat);

template <typename T, int R>
std::vector<T> matrix_to_array(
    const Eigen::Matrix<T, R, 1>& i_mat);

Eigen::SparseMatrix<double> to_real_quaternion_matrix(
    const Eigen::SparseMatrix<double>& i_real_matrix);

Eigen::Matrix<double, Eigen::Dynamic, 4> to_quaternion_matrix(
    const gsl::span<const Eigen::Vector4d> i_qvec);

std::vector<Eigen::Vector4d> to_quaternion_vector(const Eigen::VectorXd& i_vec);

template <typename T, int R>
Eigen::Map<
Eigen::Matrix<T, Eigen::Dynamic, R, Eigen::RowMajor>, Eigen::Unaligned> 
array_to_matrix(
    gsl::span<Eigen::Matrix<T, R, 1>> i_array)
{
  using namespace Eigen;
  Map<Matrix<T, Dynamic, R, RowMajor>, Unaligned> array_mask(
      &i_array[0][0], i_array.size(), R);
  return array_mask;
}

template <typename T, int R>
Eigen::Map<
const Eigen::Matrix<T, Eigen::Dynamic, R, Eigen::RowMajor>, Eigen::Unaligned> 
array_to_matrix(
    const gsl::span<const Eigen::Matrix<T, R, 1>> i_array)
{
  using namespace Eigen;
  Map<const Matrix<T, Dynamic, R, RowMajor>, Unaligned> array_mask(
      &i_array[0][0], i_array.size(), R);
  return array_mask;
}

#include "detail/flo_matrix_operation.inl"

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_MATRIX_OPERATION
