#ifndef FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

thrust::device_vector<int>
vertex_vertex_adjacency(const thrust::device_ptr<const int3> di_faces,
                        const int i_nfaces,
                        const int i_nvertices,
                        thrust::device_ptr<int> do_valence,
                        thrust::device_ptr<int> do_cumulative_valence);

thrust::device_vector<int2> adjacency_matrix_offset(
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const int> di_vertex_adjacency,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const int i_nfaces);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

