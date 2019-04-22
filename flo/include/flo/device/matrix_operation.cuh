#ifndef FLO_DEVICE_INCLUDED_MATRIX_OPERATION
#define FLO_DEVICE_INCLUDED_MATRIX_OPERATION

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void find_diagonal_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_row_offsets,
  cusp::array1d<int, cusp::device_memory>::const_view di_row_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_column_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals);

FLO_API void make_skip_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_skip_keys,
  cusp::array1d<int, cusp::device_memory>::view do_iterator_indices);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_MATRIX_OPERATION
