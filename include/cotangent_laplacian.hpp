#ifndef FLO_INCLUDED_COTANGENT_LAPLACIAN
#define FLO_INCLUDED_COTANGENT_LAPLACIAN

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_NAMESPACE_BEGIN

Eigen::SparseMatrix<double> cotangent_laplacian(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_COTANGENT_LAPLACIAN
