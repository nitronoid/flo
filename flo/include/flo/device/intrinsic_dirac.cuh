#ifndef FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
#define FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC

#include "flo/flo_internal.hpp"
#include "flo/device/quaternion_matrix.cuh"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void intrinsic_dirac_values(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_diagonal_indices,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix);

FLO_API void intrinsic_dirac(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
   cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
 cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix);

FLO_API void intrinsic_dirac(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
