#include "spin_xform.hpp"
#include "flo_matrix_operation.hpp"
#include "intrinsic_dirac.hpp"
#include "cotangent_laplacian.hpp"
#include "valence.hpp"
#include "area.hpp"
#include "similarity_xform.hpp"
#include "divergent_edges.hpp"
#include "spin_positions.hpp"

using namespace Eigen;

FLO_NAMESPACE_BEGIN

std::vector<Vector3d> spin_xform(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const double> i_rho,
    const SparseMatrix<double> i_cotangent_laplacian)
{
  // Calculate the real matrix from our quaternion edges
  auto ql = flo::to_real_quaternion_matrix(i_cotangent_laplacian);
  // Remove the final quaternion so we have a positive semi-definite matrix
  ql.conservativeResize(ql.rows() - 4, ql.cols() - 4);

  // Calculate all face areas
  auto face_area = flo::area(i_vertices, i_faces);

  // Calculate the valence of every vertex to allocate sparse matrices
  auto vertex_valence = flo::valence(i_faces);

  // Calculate the intrinsic dirac operator matrix
	auto dq = flo::intrinsic_dirac(
      i_vertices, i_faces, vertex_valence, face_area, i_rho);

  // Calculate the scaling and rotation for our spin transformation
  auto lambda = flo::similarity_xform(dq);

  // Calculate our transformed edges
	auto new_edges = flo::divergent_edges(
      i_vertices, i_faces, lambda, i_cotangent_laplacian);
  // Remove the final edge to ensure we are compatible with the sliced laplacian
  new_edges.pop_back();

  // Solve the final vertex positions
  auto new_positions = flo::spin_positions(ql, new_edges);

  return new_positions;
}

std::vector<Vector3d> spin_xform(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const double> i_rho)
{
  // Wrapper to calculate the laplacian when not already provided
  auto L = flo::cotangent_laplacian(i_vertices, i_faces);
  return spin_xform(i_vertices, i_faces, i_rho, L);
}


FLO_NAMESPACE_END

