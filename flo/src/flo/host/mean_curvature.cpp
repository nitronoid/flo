#include "flo/host/mean_curvature.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <algorithm>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

std::vector<Vector3d> mean_curvature_normal(
    const gsl::span<const Vector3d> i_vertices,
    const SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass)
{
  auto V = array_to_matrix(i_vertices);
  Map<const VectorXd> M(i_vertex_mass.data(), i_vertex_mass.size());

  VectorXd Minv = 1. / (12. * M.array());
  Matrix<double, Dynamic, 3> HN 
    = (-Minv).asDiagonal() * (2.0 * i_cotangent_laplacian * V);
  auto curvature_normal = matrix_to_array(HN);
  return curvature_normal;
}

std::vector<double> mean_curvature(
    const gsl::span<const Vector3d> i_vertices,
    const SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass)
{
  auto curvature_normals = mean_curvature_normal(
      i_vertices, i_cotangent_laplacian, i_vertex_mass);

  std::vector<double> curvature(curvature_normals.size());
  std::transform(
      curvature_normals.begin(),
      curvature_normals.end(),
      curvature.begin(),
      [](const auto& cn) 
      {
        return cn.norm(); 
      });
  return curvature;
}

std::vector<double> signed_mean_curvature(
    const gsl::span<const Vector3d> i_vertices,
    const SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass,
    const gsl::span<const Vector3d> i_normals)
{
  auto curvature_normals = mean_curvature_normal(
      i_vertices, i_cotangent_laplacian, i_vertex_mass);

  std::vector<double> curvature(curvature_normals.size());
  for (uint i = 0; i < curvature_normals.size(); ++i)
  {
    // if the angle between the unit and curvature normals is obtuse,
    // we need to flow in the opposite direction, and hence invert our sign
    auto NdotH = -i_normals[i].dot(curvature_normals[i]);
    curvature[i] = std::copysign(curvature_normals[i].norm(), std::move(NdotH));
  }
  return curvature;
}

FLO_HOST_NAMESPACE_END

