#ifndef FLO_DEVICE_INCLUDED_QUATERNION_MATRIX
#define FLO_DEVICE_INCLUDED_QUATERNION_MATRIX

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void to_quaternion_matrix(
  cusp::coo_matrix<int, real4, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix);

FLO_API void to_real_quaternion_matrix(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix);


FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_QUATERNION_MATRIX

