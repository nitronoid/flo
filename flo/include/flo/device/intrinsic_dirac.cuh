#ifndef FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
#define FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>

FLO_DEVICE_NAMESPACE_BEGIN

void intrinsic_dirac(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const thrust::device_ptr<const real> di_rho,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const thrust::device_ptr<const int2> di_entry_offset,
  const thrust::device_ptr<const int> di_cumulative_triangle_valence,
  const thrust::device_ptr<const int> di_vertex_triangle_adjacency,
  const int i_nverts,
  const int i_nfaces,
  thrust::device_ptr<int> do_diagonals,
  thrust::device_ptr<int> do_rows,
  thrust::device_ptr<int> do_columns,
  thrust::device_ptr<real4> do_values);

void to_real_quaternion_matrix(
  const thrust::device_ptr<const int> di_rows,
  const thrust::device_ptr<const int> di_columns,
  const thrust::device_ptr<const real4> di_values,
  const thrust::device_ptr<const int> di_column_size,
  const int i_nvalues,
  thrust::device_ptr<int> do_rows,
  thrust::device_ptr<int> do_columns,
  thrust::device_ptr<real> do_values);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
