#include "flo/host/vertex_normals.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/per_vertex_normals.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Matrix<real, 3, 1>> vertex_normals(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  auto V = array_to_matrix(i_vertices);
  auto F = array_to_matrix(i_faces);
  Matrix<real, Dynamic, 3> N;
  igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);

  auto normals = matrix_to_array(N);
  return normals;
}

FLO_HOST_NAMESPACE_END
