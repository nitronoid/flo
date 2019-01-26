#ifndef FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

void vertex_triangle_adjacency(
    const thrust::device_vector<int>& di_faces,
    const uint i_n_verts,
    thrust::device_vector<int>& dio_adjacency,
    thrust::device_vector<int>& dio_valence,
    thrust::device_vector<int>& dio_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY


