#ifndef FLO_INCLUDED_VERTEX_MASS
#define FLO_INCLUDED_VERTEX_MASS

#include "flo_internal.hpp"
#include <Eigen/Dense>

FLO_NAMESPACE_BEGIN

std::vector<double> vertex_mass(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_VERTEX_MASS

