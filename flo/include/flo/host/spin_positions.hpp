#ifndef FLO_HOST_INCLUDED_SPIN_POSITIONS
#define FLO_HOST_INCLUDED_SPIN_POSITIONS

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Matrix<real, 3, 1>> spin_positions(
    const Eigen::SparseMatrix<real>& i_quaternion_laplacian, 
    const gsl::span<const Eigen::Matrix<real, 4, 1>> i_quaternion_edge_matrix);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_SPIN_POSITIONS

