#ifndef FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

void vertex_triangle_adjacency(
    const thrust::device_ptr<int> dio_faces,
    const uint i_nfaces,
    const uint i_nverts,
    thrust::device_ptr<int> do_adjacency,
    thrust::device_ptr<int> do_valence,
    thrust::device_ptr<int> do_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY


