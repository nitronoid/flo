#ifndef FLO_INCLUDED_SPIN_POSITIONS
#define FLO_INCLUDED_SPIN_POSITIONS

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_NAMESPACE_BEGIN

std::vector<Eigen::Vector3d> spin_positions(
    const Eigen::SparseMatrix<double>& i_quaternion_laplacian, 
    const gsl::span<const Eigen::Vector4d> i_quaternion_edge_matrix);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_SPIN_POSITIONS

