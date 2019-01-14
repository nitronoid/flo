#ifndef FLO_INCLUDED_SPIN_XFORM
#define FLO_INCLUDED_SPIN_XFORM

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

std::vector<Eigen::Vector3d> spin_xform(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const double> i_rho,
    const Eigen::SparseMatrix<double> i_cotangent_laplacian);

std::vector<Eigen::Vector3d> spin_xform(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const double> i_rho);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_SPIN_XFORM
