#ifndef FLO_HOST_INCLUDED_COTANGENT_LAPLACIAN
#define FLO_HOST_INCLUDED_COTANGENT_LAPLACIAN

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API Eigen::SparseMatrix<double> cotangent_laplacian(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_COTANGENT_LAPLACIAN
