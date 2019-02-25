#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

void cotangent_laplacian(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const thrust::device_ptr<const int2> di_entry_offset,
  const int i_nverts,
  const int i_nfaces,
  const int i_total_valence,
  thrust::device_ptr<int> do_diagonals,
  thrust::device_ptr<int> do_rows,
  thrust::device_ptr<int> do_columns,
  thrust::device_ptr<real> do_values);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
