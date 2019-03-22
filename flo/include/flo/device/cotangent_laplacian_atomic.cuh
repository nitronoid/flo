#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

void cotangent_laplacian(
  cusp::array1d<real3, cusp::device_memory>::const_view di_vertices,
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<int2, cusp::device_memory>::const_view di_entry_offset,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view
    do_cotangent_laplacian);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
