#ifndef FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
#define FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void intrinsic_dirac(
  cusp::array1d<real3, cusp::device_memory>::const_view di_vertices,
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
  cusp::array1d<int2, cusp::device_memory>::const_view di_entry_offset,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_cumulative_triangle_valence,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix);

FLO_API void to_real_quaternion_matrix(
  cusp::coo_matrix<int, real4, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC