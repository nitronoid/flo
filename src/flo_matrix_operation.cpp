#include "flo_matrix_operation.hpp"
#include "flo_quaternion_operation.hpp"
#include <iostream>

using namespace Eigen;

FLO_NAMESPACE_BEGIN

SparseMatrix<double> to_real_quaternion_matrix(
    const SparseMatrix<double>& i_real_matrix)
{
  SparseMatrix<double> quat_matrix(
      i_real_matrix.rows()*4, i_real_matrix.cols()*4);

  Eigen::Matrix<int, Dynamic, 1> dim(i_real_matrix.cols() * 4, 1);
  for (uint i = 0u; i < i_real_matrix.cols(); ++i)
  {
    auto num =  4 * i_real_matrix.col(i).nonZeros();
    dim(i * 4 + 0) = num;
    dim(i * 4 + 1) = num;
    dim(i * 4 + 2) = num;
    dim(i * 4 + 3) = num;
  }
  quat_matrix.reserve(dim);

  using iter_t = SparseMatrix<double>::InnerIterator;
  for (uint i = 0u; i < i_real_matrix.outerSize(); ++i)
  {
    for (iter_t it(i_real_matrix, i); it; ++it)
    {
      auto r = it.row();
      auto c = it.col();

      Vector4d real_quat(0.f, 0.f, 0.f, it.value());
      auto block = quat_to_block(real_quat);
      insert_block_sparse(block, quat_matrix, r, c);
    }
  }
  return quat_matrix;
}

Matrix<double, Dynamic, 4> to_quaternion_matrix(
    const gsl::span<const Vector4d> i_qvec)
{
  // Real matrix of No. Quats * 4 x 4
  Matrix<double, Dynamic, 4> mat(i_qvec.size() * 4, 4);

  for (uint i = 0; i < i_qvec.size(); ++i)
  {
    auto block = quat_to_block(i_qvec[i]);
    insert_block_dense(block, mat, i, 0);
  }

  return mat;
}

std::vector<Vector4d> to_quaternion_vector(const VectorXd& i_vec)
{
  std::vector<Vector4d> qmat(i_vec.rows() / 4);
  for (uint r = 0u; r < qmat.size(); ++r)
  {
    qmat[r] = Vector4d({i_vec(r*4 + 1, 0),
                        i_vec(r*4 + 2, 0),
                        i_vec(r*4 + 3, 0),
                        i_vec(r*4 + 0, 0)});
  }
  return qmat;
}

FLO_NAMESPACE_END
