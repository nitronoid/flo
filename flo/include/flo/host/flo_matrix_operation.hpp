#ifndef FLO_HOST_INCLUDED_MATRIX_OPERATION
#define FLO_HOST_INCLUDED_MATRIX_OPERATION

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

template <int R, int C>
FLO_API void insert_block_sparse(
    const Eigen::Matrix<real, R, C>& i_block,
    Eigen::SparseMatrix<real>& i_mat,
    uint i_x,
    uint i_y);

template <int R, int C, int IR, int IC>
FLO_API void insert_block_dense(
    const Eigen::Matrix<real, R, C>& i_block,
    Eigen::Matrix<real, IR, IC>& i_mat,
    uint i_x,
    uint i_y);

template <typename T, int R, int C>
FLO_API std::vector<Eigen::Matrix<T, C, 1>> matrix_to_array(
    const Eigen::Matrix<T, R, C>& i_mat);

template <typename T, int R>
FLO_API std::vector<T> matrix_to_array(
    const Eigen::Matrix<T, R, 1>& i_mat);

FLO_API Eigen::SparseMatrix<real> to_real_quaternion_matrix(
    const Eigen::SparseMatrix<real>& i_real_matrix);

FLO_API Eigen::Matrix<real, Eigen::Dynamic, 4> to_quaternion_matrix(
    const gsl::span<const Eigen::Matrix<real, 4, 1>> i_qvec);

FLO_API std::vector<Eigen::Matrix<real, 4, 1>> to_quaternion_vector(const Eigen::Matrix<real, Eigen::Dynamic, 1>& i_vec);

template <typename T>
FLO_API Eigen::Map<
Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Unaligned> 
array_to_matrix(gsl::span<T> i_array)
{
  using namespace Eigen;
  Map<Matrix<T, Eigen::Dynamic, 1>, Unaligned> array_mask(
      i_array.data(), i_array.size(), 1);
  return array_mask;
}

template <typename T>
FLO_API Eigen::Map<
const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Unaligned> 
array_to_matrix(const gsl::span<const T> i_array)
{
  using namespace Eigen;
  Map<const Matrix<T, Eigen::Dynamic, 1>, Unaligned> array_mask(
      i_array.data(), i_array.size(), 1);
  return array_mask;
}

template <typename T, int R>
FLO_API Eigen::Map<
Eigen::Matrix<T, Eigen::Dynamic, R, Eigen::RowMajor>, Eigen::Unaligned> 
array_to_matrix(
    gsl::span<Eigen::Matrix<T, R, 1>> i_array)
{
  using namespace Eigen;
  Map<Matrix<T, Eigen::Dynamic, R, RowMajor>, Unaligned> array_mask(
      &i_array[0][0], i_array.size(), R);
  return array_mask;
}

template <typename T, int R>
FLO_API Eigen::Map<
const Eigen::Matrix<T, Eigen::Dynamic, R, Eigen::RowMajor>, Eigen::Unaligned> 
array_to_matrix(
    const gsl::span<const Eigen::Matrix<T, R, 1>> i_array)
{
  using namespace Eigen;
  Map<const Matrix<T, Eigen::Dynamic, R, RowMajor>, Unaligned> array_mask(
      &i_array[0][0], i_array.size(), R);
  return array_mask;
}

#include "flo/host/flo_matrix_operation.inl"//template definitions

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_MATRIX_OPERATION
