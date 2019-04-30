#ifndef FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES
#define FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @brief Computes the row, column, entry and diagonal indices for an adjacency 
/// matrix sparsity pattern
/// @param di_faces, A column major matrix of face vertex indices
/// @param di_adjacency_keys, An array of vertex vertex adjacency keys
/// @param di_adjacency_keys, An array of vertex vertex adjacency 
/// @param di_adjacency_keys, An array of vertex vertex cumulative valence 
/// @param do_entry_indices, An output array of the entry indices
/// @param do_diagonal_indices, An output array of the diagonal entry indices
/// @param do_row_indices, An output array of the row indices
/// @param do_column_indices, An output array of the column indices
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

/// @brief Computes the diagonal indices for a matrix
/// @param di_row_offsets, An array of the row offsetss (CSR format)
/// @param di_row_indices, An array of the row indices (COO format)
/// @param di_column_indices, An array of the column indices (COO format)
/// @param do_diagonals, An output array of the diagonal entry indices
FLO_API void find_diagonal_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_row_offsets,
  cusp::array1d<int, cusp::device_memory>::const_view di_row_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_column_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals);

/// @brief Computes the permutation indices required to skip the supplied keys
/// @param di_skip_keys, An array of the indices to skip
/// @param do_iterator_indices, The permutation indices
FLO_API void make_skip_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_skip_keys,
  cusp::array1d<int, cusp::device_memory>::view do_iterator_indices);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES


