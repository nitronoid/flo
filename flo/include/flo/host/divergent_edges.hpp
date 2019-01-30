#ifndef FLO_HOST_INCLUDED_DIVERGENT_EDGES
#define FLO_HOST_INCLUDED_DIVERGENT_EDGES

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Matrix<real, 4, 1>> divergent_edges(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const Eigen::Matrix<real, 4, 1>> i_lambda,
    const Eigen::SparseMatrix<real> i_cotangent_laplacian);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_DIVERGENT_EDGES

