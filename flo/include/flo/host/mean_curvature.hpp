#ifndef FLO_HOST_INCLUDED_MEAN_CURVATURE
#define FLO_HOST_INCLUDED_MEAN_CURVATURE

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Matrix<real, 3, 1>> mean_curvature_normal(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const Eigen::SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass);

FLO_API std::vector<real> mean_curvature(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const Eigen::SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass);

FLO_API std::vector<real> signed_mean_curvature(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const Eigen::SparseMatrix<real>& i_cotangent_laplacian,
    const gsl::span<const real> i_vertex_mass,
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_normals);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_MEAN_CURVATURE

