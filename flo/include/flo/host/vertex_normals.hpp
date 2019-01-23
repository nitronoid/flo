#ifndef FLO_HOST_INCLUDED_VERTEX_NORMALS
#define FLO_HOST_INCLUDED_VERTEX_NORMALS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

std::vector<Eigen::Vector3d> vertex_normals(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_VERTEX_NORMALS

