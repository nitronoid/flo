#ifndef FLO_INCLUDED_AREA
#define FLO_INCLUDED_AREA

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_NAMESPACE_BEGIN

std::vector<double> area(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_AREA
