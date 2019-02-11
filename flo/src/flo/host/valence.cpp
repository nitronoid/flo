#include "flo/host/valence.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <Eigen/Sparse>
#include <igl/adjacency_matrix.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<int> valence(const gsl::span<const Vector3i> i_faces)
{
  auto F = array_to_matrix(i_faces);
  SparseMatrix<int> A;
  igl::adjacency_matrix(F, A);
  std::vector<int> degree(A.cols());
  for (uint i = 0; i < A.cols(); ++i)
  {
    degree[i] = A.col(i).nonZeros();
  }
  return degree;
}

FLO_HOST_NAMESPACE_END
