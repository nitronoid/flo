#ifndef FLO_DEVICE_INCLUDED_MEAN_CURVATURE
#define FLO_DEVICE_INCLUDED_MEAN_CURVATURE

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void mean_curvature_normal(
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view di_vertices,
  cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_cotangent_laplacian,
  cusp::array1d<real, cusp::device_memory>::const_view di_vertex_mass,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view do_mean_curvature_normals);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_MEAN_CURVATURE

