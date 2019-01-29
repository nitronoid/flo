#ifndef FLO_HOST_INCLUDED_MEAN_CURVATURE
#define FLO_HOST_INCLUDED_MEAN_CURVATURE

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Vector3d> mean_curvature_normal(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const Eigen::SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass);

FLO_API std::vector<double> mean_curvature(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const Eigen::SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass);

FLO_API std::vector<double> signed_mean_curvature(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const Eigen::SparseMatrix<double>& i_cotangent_laplacian,
    const gsl::span<const double> i_vertex_mass,
    const gsl::span<const Eigen::Vector3d> i_normals);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_MEAN_CURVATURE

