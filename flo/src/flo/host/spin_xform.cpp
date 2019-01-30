#include "flo/host/spin_xform.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/valence.hpp"
#include "flo/host/area.hpp"
#include "flo/host/similarity_xform.hpp"
#include "flo/host/divergent_edges.hpp"
#include "flo/host/spin_positions.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Matrix<real, 3, 1>> spin_xform(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const real> i_rho,
    const SparseMatrix<real> i_cotangent_laplacian)
{
  // Calculate the real matrix from our quaternion edges
  auto ql = to_real_quaternion_matrix(i_cotangent_laplacian);
  // Remove the final quaternion so we have a positive semi-definite matrix
  ql.conservativeResize(ql.rows() - 4, ql.cols() - 4);

  // Calculate all face areas
  auto face_area = area(i_vertices, i_faces);

  // Calculate the valence of every vertex to allocate sparse matrices
  auto vertex_valence = valence(i_faces);

  // Calculate the intrinsic dirac operator matrix
	auto dq = intrinsic_dirac(
      i_vertices, i_faces, vertex_valence, face_area, i_rho);

  // Calculate the scaling and rotation for our spin transformation
  auto lambda = similarity_xform(dq);

  // Calculate our transformed edges
	auto new_edges = divergent_edges(
      i_vertices, i_faces, lambda, i_cotangent_laplacian);
  // Remove the final edge to ensure we are compatible with the sliced laplacian
  new_edges.pop_back();

  // Solve the final vertex positions
  auto new_positions = spin_positions(ql, new_edges);

  return new_positions;
}

FLO_API std::vector<Matrix<real, 3, 1>> spin_xform(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const real> i_rho)
{
  // Wrapper to calculate the laplacian when not already provided
  auto L = cotangent_laplacian(i_vertices, i_faces);
  return spin_xform(i_vertices, i_faces, i_rho, L);
}


FLO_HOST_NAMESPACE_END

