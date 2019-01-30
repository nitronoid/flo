#include "flo/host/mean_curvature.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <algorithm>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Matrix<real, 3, 1>> mean_curvature_normal(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass)
{
  auto V = array_to_matrix(i_vertices);
  Map<const Matrix<real, Dynamic, 1>> M(i_vertex_mass.data(), i_vertex_mass.size());

  Matrix<real, Dynamic, 1> Minv = 1. / (12. * M.array());
  Matrix<real, Dynamic, 3> HN 
    = (-Minv).asDiagonal() * (2.0 * i_cotangent_laplacian * V);
  auto curvature_normal = matrix_to_array(HN);
  return curvature_normal;
}

FLO_API std::vector<real> mean_curvature(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass)
{
  auto curvature_normals = mean_curvature_normal(
      i_vertices, i_cotangent_laplacian, i_vertex_mass);

  std::vector<real> curvature(curvature_normals.size());
  std::transform(
      curvature_normals.begin(),
      curvature_normals.end(),
      curvature.begin(),
      [](const Matrix<real, 3, 1>& cn) 
      {
        return cn.norm(); 
      });
  return curvature;
}

FLO_API std::vector<real> signed_mean_curvature(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass,
    const gsl::span<const Matrix<real, 3, 1>> i_normals)
{
  auto curvature_normals = mean_curvature_normal(
      i_vertices, i_cotangent_laplacian, i_vertex_mass);

  std::vector<real> curvature(curvature_normals.size());
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

