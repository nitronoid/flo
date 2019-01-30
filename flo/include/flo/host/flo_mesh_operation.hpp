#ifndef FLO_HOST_INCLUDED_MESH_OPERATION
#define FLO_HOST_INCLUDED_MESH_OPERATION

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API void remove_mean(gsl::span<Eigen::Matrix<real, 4, 1>> io_positions);

FLO_API void normalize_positions(gsl::span<Eigen::Matrix<real, 4, 1>> io_positions);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_MESH_OPERATION
