#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @brief Computes the cotangent laplacian matrix values
/// @param di_vertices A row major matrix of vertex positions
/// @param di_faces A row major matrix of face vertex indices
/// @param di_entry_indices A row major matrix of lower/upper output indices
/// @param di_diagonals An array of diagonal entry output indices
/// @param do_cotangent_laplacian A sparse coo_matrix to store the result
FLO_API void cotangent_laplacian_values(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

/// @brief Computes the cotangent laplacian matrix values and keys
/// @param di_vertices A row major matrix of vertex positions
/// @param di_faces A row major matrix of face vertex indices
/// @param di_adjacency_keys An array of vertex vertex adjacency keys
/// @param di_adjacency An array of vertex vertex adjacency
/// @param di_cumulative_valence An array of vertex vertex cumulative valence
/// @param do_entry_indices A row major matrix of lower/upper output indices
/// @param do_diagonals An array of diagonal entry output indices
/// @param do_cotangent_laplacian A sparse coo_matrix to store the result
FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

/// @brief Computes the cotangent laplacian matrix values and keys, and performs
/// allocation of the key arrays
/// @param di_vertices A row major matrix of vertex positions
/// @param di_faces A row major matrix of face vertex indices
/// @param di_adjacency_keys An array of vertex vertex adjacency keys
/// @param di_adjacency An array of vertex vertex adjacency
/// @param di_cumulative_valence An array of vertex vertex cumulative valence
/// @param do_cotangent_laplacian A sparse coo_matrix to store the result
FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN
