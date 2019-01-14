#ifndef FLO_INCLUDED_DIVERGENT_EDGES
#define FLO_INCLUDED_DIVERGENT_EDGES

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

std::vector<Eigen::Vector4d> divergent_edges(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const Eigen::Vector4d> i_lambda,
    const Eigen::SparseMatrix<double> i_cotangent_laplacian);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_DIVERGENT_EDGES

