#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/cotmatrix.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API SparseMatrix<real> cotangent_laplacian(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  auto V = array_to_matrix(i_vertices);
  auto F = array_to_matrix(i_faces);
  SparseMatrix<real> L;
  igl::cotmatrix(V, F, L);
  // Convert to positive semi-definite
  L = (-L.eval());
  return L;
}

FLO_HOST_NAMESPACE_END

