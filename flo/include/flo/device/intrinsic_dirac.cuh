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
  const int i_nverts,
  const int i_nfaces,
  const int i_total_valence,
  thrust::device_ptr<int> do_diagonals,
  thrust::device_ptr<int> do_rows,
  thrust::device_ptr<int> do_columns,
  thrust::device_ptr<real4> do_values);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_INTRINSIC_DIRAC
