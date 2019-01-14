#include <iostream>
#include <igl/invert_diag.h>

#include "cotangent_laplacian.hpp"
#include "load_mesh.hpp"
#include "flo_matrix_operation.hpp"
#include "spin_xform.hpp"
#include "vertex_normals.hpp"
#include "area.hpp"

using namespace Eigen;
using namespace flo;

//std::vector<double> mean_curvature(
//    const gsl::span<const Vector3d> i_vertices,
//    const SparseMatrix<double>& i_cotangent_laplacian,
//    const gsl::span<const double> i_vertex_mass)
//{
//  auto V = array_to_matrix(i_vertices);
//  Map<const VectorXd> M(i_vertex_mass.data(), i_vertex_mass.size());
//
//  Matrix<double, Dynamic, 3> HN(i_vertices.size(), 3); 
//  HN = (2. * i_cotangent_laplacian * V);
//
//  HN.col(0).array() /= (12. *M.array());
//  HN.col(1).array() /= (12. *M.array());
//  HN.col(2).array() /= (12. *M.array());
//  std::cout<<HN<<'\n';
//  VectorXd H = HN.rowwise().norm();
//
//  auto curvature = matrix_to_array(H);
//  return curvature;
//}

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

std::vector<double> mean_curvature(
    const gsl::span<const Vector3d> i_vertices,
    const SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass)
{
  auto V = array_to_matrix(i_vertices);
  Map<const VectorXd> M(i_vertex_mass.data(), i_vertex_mass.size());

  VectorXd Minv = -1. / 12. * M.array();
  MatrixXd HN = Minv.asDiagonal() * (2 * i_cotangent_laplacian * V);
  VectorXd H = HN.rowwise().norm();

  auto curvature = matrix_to_array(H);
  return curvature;
}

int main()
{
  auto surf = flo::load_mesh("foo.obj");
  auto normals = flo::vertex_normals(surf.vertices, surf.faces);
  //for (const auto& n : normals)
  //  std::cout<<n<<'\n';

  // Calculate the cotangent laplacian for our mesh
  auto L = flo::cotangent_laplacian(surf.vertices, surf.faces);
  auto mass = vertex_mass(surf.vertices, surf.faces);

  auto mc = mean_curvature(surf.vertices, L, mass);
  for (const auto& c : mc)
    std::cout<<c<<'\n';

  // Calculate change in mean curvature half density
  std::vector<double> rho(surf.n_vertices(), -0.0666667f);

  auto new_positions = flo::spin_xform(surf.vertices, surf.faces, rho, L);
  for (const auto& p : new_positions)
    std::cout<<p<<'\n';

  return 0;
}
