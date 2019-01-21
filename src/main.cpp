#include <iostream>
#include <igl/invert_diag.h>
#include <igl/writeOBJ.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>

#include "cotangent_laplacian.hpp"
#include "load_mesh.hpp"
#include "flo_matrix_operation.hpp"
#include "flo_quaternion_operation.hpp"
#include "spin_xform.hpp"
#include "vertex_normals.hpp"
#include "area.hpp"

using namespace Eigen;
using namespace flo;

std::vector<double> mean_curvature(
    const gsl::span<const Vector3d> i_vertices,
    const SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass,
    const gsl::span<const Vector3d> i_normals)
{
  auto V = array_to_matrix(i_vertices);
  auto N = array_to_matrix(i_normals);
  Map<const VectorXd> M(i_vertex_mass.data(), i_vertex_mass.size());

  Matrix<double, Dynamic, 3> HN(i_vertices.size(), 3); 
  HN = (i_cotangent_laplacian * V);

  const auto coeff = 1. / 12.;
  HN.col(0).array() /= (12. *M.array() * N.col(0).array());
  HN.col(1).array() /= (12. *M.array() * N.col(1).array());
  HN.col(2).array() /= (12. *M.array() * N.col(2).array());
  VectorXd H = HN.rowwise().mean();
  
  H *= 0.5;
  //std::cout<<"count: "<<count<<'\n';
  //std::cout<<HN<<'\n';

  auto curvature = matrix_to_array(H);
  return curvature;
}

std::vector<double> vertex_mass(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  std::vector<double> mass(i_vertices.size());
  auto face_area = area(i_vertices, i_faces);

  // For every face
  for (uint i = 0; i < i_faces.size(); ++i)
  {
    const auto& f = i_faces[i];
    constexpr auto third = 1.f / 3.f;
    auto thirdArea = face_area[i] * third;

    mass[f(0)] += thirdArea;
    mass[f(1)] += thirdArea;
    mass[f(2)] += thirdArea;
  }

  return mass;
}

template <typename T, typename F>
MatrixXd build_constraint_basis(const T& i_vectors, F&& i_inner_product)
{
  // Normalize is defined using a self inner product
  auto normalize = [&](const auto& x) {
    return x.array() / std::sqrt(i_inner_product(x, x)(0, 0));
  };
  const auto vlen = i_vectors.size();
  std::cout<<vlen<<'\n';
  const auto n_vec = sizeof(i_vectors[0]) / sizeof(double);

  // Map our input vectors as a matrix
  auto matrix = array_to_matrix(gsl::make_span(i_vectors));
  //std::cout<<matrix<<'\n';

  // Declare and allocate space for our final basis matrix
  MatrixXd basis(vlen, n_vec);

  // The first vector is filled with normalized ones
  basis.col(0) = normalize(matrix.col(0));
  // Gramm Schmit process
  for (uint i = 1u; i < n_vec; ++i)
  {
    basis.col(i) =
      matrix.col(i) -
      i_inner_product(matrix.col(i), basis.col(0)) * basis.col(0);
    for (uint k = 1u; k < i; ++k)
    {
      basis.col(i) -= 
        i_inner_product(basis.col(i), basis.col(k)) * basis.col(k);
    }
    basis.col(i) = normalize(basis.col(i).eval());
  }

  return basis;
}

template <typename D, typename I>
void project_constraints(gsl::span<double> io_curvature,
                      const float i_step,
                      const MatrixXd& i_basis,
                      D&& i_direction,
                      I&& i_inner_product)
{
  const auto vlen = io_curvature.size();
  const auto n_vecs = i_basis.cols();
  auto kappa = array_to_matrix(io_curvature);
  VectorXd kappa_dot = i_direction(kappa);

  VectorXd accum(vlen);
  accum.setZero();
  for (uint i = 0u; i < n_vecs; ++i)
  {
    accum += i_inner_product(kappa_dot, i_basis.col(i))(0, 0) * i_basis.col(i);
  }
  kappa_dot -= accum;

  kappa = i_step * kappa_dot;
}

//std::vector<double> mean_curvature(
//    const gsl::span<const Vector3d> i_vertices,
//    const SparseMatrix<double>& i_cotangent_laplacian,
//    const gsl::span<const double> i_vertex_mass)
//{
//  auto V = array_to_matrix(i_vertices);
//  Map<const VectorXd> M(i_vertex_mass.data(), i_vertex_mass.size());
//
//  VectorXd Minv = -1. / 4. * M.array();
//  MatrixXd HN = Minv.asDiagonal() * (2.f *i_cotangent_laplacian * V);
//  VectorXd H = -HN.rowwise().norm();
//
//  auto curvature = matrix_to_array(H);
//  return curvature;
//}

int main()
{
  auto surf = flo::load_mesh("foo.obj");

  for (int iter = 0; iter < 3; ++iter)
  {
  auto normals = flo::vertex_normals(surf.vertices, surf.faces);
  for (const auto& n : normals)
    std::cout<<n<<'\n';
  std::cout<<normals.size()<<'\n';

  // Calculate the cotangent laplacian for our mesh
  auto L = flo::cotangent_laplacian(surf.vertices, surf.faces);
  auto mass = vertex_mass(surf.vertices, surf.faces);
  auto M = array_to_matrix(gsl::make_span(mass));

  //std::cout<<"yo1\n";
  std::vector<Vector4d> constraints; constraints.reserve(normals.size());
  for (int i = 0; i < normals.size(); ++i)
    constraints.emplace_back(1., normals[i].x(), normals[i].y(), normals[i].z()); 
  std::cout<<"yo2\n";
  const auto ip = [&](const auto& x, const auto& y) {
    return x.transpose() * M.asDiagonal() * y;
  };
  auto basis = build_constraint_basis(constraints, ip);
  //std::cout<<"yo3\n";

  auto mc = mean_curvature(surf.vertices, L, mass, normals);
  const auto di = [](const auto& matrix) noexcept 
  {                               
    return -1.f * matrix;        
  };
  project_constraints(mc, 0.95f, basis, di, ip);
  //for (const auto& c : mc)
  //  std::cout<<c<<'\n';

  surf.vertices = flo::spin_xform(surf.vertices, surf.faces, mc, L);
  }
  
  //for (const auto& p : new_positions)
  //  std::cout<<p<<'\n';

  auto V = array_to_matrix(gsl::make_span(surf.vertices));
  auto F = array_to_matrix(gsl::make_span(surf.faces));

  igl::writeOBJ("bar.obj", V, F);

  return 0;
}
