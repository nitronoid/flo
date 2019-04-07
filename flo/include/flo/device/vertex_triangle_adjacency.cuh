#ifndef FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void vertex_triangle_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  thrust::device_ptr<int> dio_temporary_storage,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

