#ifndef FLO_INCLUDED_MESH_OPERATION
#define FLO_INCLUDED_MESH_OPERATION

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

void remove_mean(gsl::span<Eigen::Vector4d> io_positions);

void normalize_positions(gsl::span<Eigen::Vector4d> io_positions);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_MESH_OPERATION
