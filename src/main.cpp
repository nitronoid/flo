#include <iostream>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>

#include "cotangent_laplacian.hpp"
#include "mean_curvature.hpp"
#include "load_mesh.hpp"
#include "flo_matrix_operation.hpp"
#include "flo_quaternion_operation.hpp"
#include "spin_xform.hpp"
#include "vertex_normals.hpp"
#include "vertex_mass.hpp"
#include "area.hpp"

using namespace Eigen;
using namespace flo;


template <typename F>
std::vector<double> orthonormalize(
    const gsl::span<const double> i_vectors, 
    const uint i_num_vectors, 
    F&& i_inner_product)
{
  // Normalize is defined using a self inner product
  auto normalize = [&](const auto& x) {
    return x.array() / std::sqrt(i_inner_product(x, x));
  };
  const auto vlen = i_vectors.size() / i_num_vectors;

  // Map our input vectors as a matrix
  Map<const MatrixXd> v(i_vectors.data(), vlen, i_num_vectors); 
  //std::cout<<matrix<<'\n';

  // Declare and allocate space for our final basis matrix
  std::vector<double> basis(vlen * i_num_vectors);
  Map<MatrixXd> u(basis.data(), vlen, i_num_vectors); 

  // The first u0 is v0 normalized
  u.col(0) = normalize(v.col(0));
  // Gramm Schmit process
  for (uint i = 1u; i < i_num_vectors; ++i)
  {
    u.col(i) = v.col(i) - i_inner_product(v.col(i), u.col(0)) * u.col(0);
    for (uint k = 1u; k < i; ++k)
    {
      u.col(i) -= i_inner_product(u.col(i), u.col(k)) * u.col(k);
    }
    u.col(i) = normalize(u.col(i).eval());
  }

  return basis;
}

template <typename I>
std::vector<double> project_constraint_basis(
      const gsl::span<const double> i_curvature,
      const gsl::span<const double> i_basis, 
      const uint i_num_constraints, 
      I&& i_inner_product)
{
  const auto vlen = i_curvature.size();
  std::vector<double> projected(i_curvature.begin(), i_curvature.end());
  auto kappa_dot = array_to_matrix(gsl::make_span(projected));
  Map<const MatrixXd> u(i_basis.data(), vlen, i_num_constraints); 

  // Subtract the projected curvature from the un-projected
  for (uint i = 0u; i < i_num_constraints; ++i)
  {
    kappa_dot -= i_inner_product(kappa_dot, u.col(i)) * u.col(i);
  }

  return projected;
}

void forward_euler(gsl::span<double> i_x,
                   const gsl::span<const double> i_dx,
                   const double i_t)
{
  std::transform(i_x.begin(), i_x.end(), i_dx.begin(), i_x.begin(),
                 [i_t](auto x, auto dx)
                 {
                   return x + dx * i_t;
                 });
}

int main()
{
  auto surf = flo::load_mesh("foo.obj");

  for (int iter = 0; iter < 10; ++iter)
  {
  // Calculate smooth vertex normals
  auto normals = flo::vertex_normals(surf.vertices, surf.faces);

  // Calculate the cotangent laplacian for our mesh
  auto L = flo::cotangent_laplacian(surf.vertices, surf.faces);
  // Calculate the vertex masses for our mesh 
  auto mass = vertex_mass(surf.vertices, surf.faces);
  // Map the masses to a vector (to use as a diagonal-matrix)
  auto M = array_to_matrix(gsl::make_span(mass));

  const uint n_verts = surf.n_vertices();
  const uint n_constraints = 4;
  // Build our constraints {1, N.x, N.y, N.z}
  std::vector<double> constraints(normals.size() * n_constraints);
  for (int i = 0; i < normals.size(); ++i)
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
  auto mc = signed_mean_curvature(surf.vertices, L, mass, normals);
  // Apply our flow direction to the the mean curvature half density
  std::transform(mc.begin(), mc.end(), mc.begin(), [](auto x) { return -x; });
  // project the constraints on to our mean curvature
  auto projected = project_constraint_basis(mc, basis, n_constraints, ip);
  // take a time step
  forward_euler(mc, projected, 0.95);

  // spin transform using our change in mean curvature half-density
  surf.vertices = flo::spin_xform(surf.vertices, surf.faces, mc, L);
  }

  auto V = array_to_matrix(gsl::make_span(surf.vertices));
  auto F = array_to_matrix(gsl::make_span(surf.faces));

  igl::writeOBJ("bar.obj", V, F);

  return 0;
}
