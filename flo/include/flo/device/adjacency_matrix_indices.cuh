#ifndef FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES
#define FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void adjacency_matrix_indices(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::array1d<int, cusp::device_memory>::view do_row_indices,
  cusp::array1d<int, cusp::device_memory>::view do_column_indices,
  thrust::device_ptr<void> dio_temp);

FLO_API void find_diagonal_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_row_offsets,
  cusp::array1d<int, cusp::device_memory>::const_view di_row_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_column_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals);

FLO_API void make_skip_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_skip_keys,
  cusp::array1d<int, cusp::device_memory>::view do_iterator_indices);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES


