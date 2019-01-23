#include "flo/host/willmore_flow.hpp"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/mean_curvature.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/spin_xform.hpp"
#include "flo/host/vertex_normals.hpp"
#include "flo/host/vertex_mass.hpp"
#include "flo/host/orthonormalize.hpp"
#include "flo/host/project_basis.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

std::vector<Vector3d> willmore_flow(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    nonstd::function_ref<
    void(gsl::span<double> x, const gsl::span<const double> dx)> i_integrator)
{
  // Calculate smooth vertex normals
  auto normals = vertex_normals(i_vertices, i_faces);

  // Calculate the cotangent laplacian for our mesh
  auto L = cotangent_laplacian(i_vertices, i_faces);
  // Calculate the vertex masses for our mesh 
  auto mass = vertex_mass(i_vertices, i_faces);
  // Map the masses to a vector (to use as a diagonal-matrix)
  auto M = array_to_matrix(gsl::make_span(mass));

  const uint n_verts = i_vertices.size();
  const uint n_constraints = 4;
  // Build our constraints {1, N.x, N.y, N.z}
  std::vector<double> constraints(normals.size() * n_constraints);
  for (uint i = 0; i < normals.size(); ++i)
  {
    constraints[i + n_verts * 0] = 1.0;
    constraints[i + n_verts * 1] = normals[i].x();
    constraints[i + n_verts * 2] = normals[i].y();
    constraints[i + n_verts * 3] = normals[i].z();
  }

  // Declare an immersed inner-product using the mass matrix
  const auto ip = [&M](const auto& x, const auto& y) -> double {
    auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
    return single_mat(0,0);
  };
  // Build a constraint basis using the Gram–Schmidt process
  auto basis = orthonormalize(constraints, n_constraints, ip);

  // Calculate the signed mean curvature based on our vertex normals
  auto mc = signed_mean_curvature(i_vertices, L, mass, normals);
  // Apply our flow direction to the the mean curvature half density
  std::transform(mc.begin(), mc.end(), mc.begin(), [](auto x) { return -x; });
  // project the constraints on to our mean curvature
  auto projected = project_basis(mc, basis, n_constraints, ip);
  // take a time step
  i_integrator(mc, projected);

  // spin transform using our change in mean curvature half-density
  auto new_vertices = spin_xform(i_vertices, i_faces, mc, L);
  return new_vertices;
}

FLO_HOST_NAMESPACE_END

