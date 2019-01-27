#ifndef FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

void vertex_triangle_adjacency(
    const gsl::span<const Eigen::Vector3i> i_faces,
    const uint i_nverts,
    gsl::span<int> o_adjacency,
    gsl::span<int> o_valence,
    gsl::span<int> o_cumulative_valence);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_VERTEX_TRIANGLE_ADJACENCY



