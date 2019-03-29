#include "flo/device/cotangent_laplacian_blas.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <cusp/elementwise.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void
edges(cusp::array1d<real, cusp::device_memory>::const_view di_vertices,
      cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
      cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
      cusp::array1d<real, cusp::device_memory>::view do_edges)
{
  const int nvertices = di_vertices.size() / 3;
  // Zip iterator for each vertex column
  const auto vertex_it = thrust::make_zip_iterator(
    thrust::make_tuple(di_vertices.begin() + nvertices * 0,
                       di_vertices.begin() + nvertices * 1,
                       di_vertices.begin() + nvertices * 2));

  const int nedges = do_edges.size() / 3;
  // Zip iterator for each edge column
  const auto edge_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_edges.begin() + nedges * 0,
                       do_edges.begin() + nedges * 1,
                       do_edges.begin() + nedges * 2));

  // Copy using the second edge vertex
  thrust::gather(di_adjacency.begin(), di_adjacency.end(), vertex_it, edge_it);

  // Subtract the first vertex to produce an edge vector
  thrust::transform(
    edge_it,
    edge_it + nedges,
    thrust::make_permutation_iterator(vertex_it, di_adjacency_keys.begin()),
    edge_it,
    [] __device__(const thrust::tuple<real, real, real>& lhs,
                  const thrust::tuple<real, real, real>& rhs) {
      return thrust::make_tuple(lhs.get<0>() - rhs.get<0>(),
                                lhs.get<1>() - rhs.get<1>(),
                                lhs.get<2>() - rhs.get<2>());
    });
}

void norms(cusp::array1d<real, cusp::device_memory>::const_view di_vectors,
           cusp::array1d<real, cusp::device_memory>::view do_norms)
{
  const int nvectors = di_vectors.size() / 3;
  // Vector views over each column of the edge array
  const auto vec_x = di_vectors.subarray(nvectors * 0, nvectors);
  const auto vec_y = di_vectors.subarray(nvectors * 1, nvectors);
  const auto vec_z = di_vectors.subarray(nvectors * 2, nvectors);

  const auto vec_it = thrust::make_zip_iterator(
    thrust::make_tuple(vec_x.begin(), vec_y.begin(), vec_z.begin()));

  // Square the edge X, Y and Z components and dump their sum in the norm array
  thrust::transform(vec_it,
                    vec_it + nvectors,
                    do_norms.begin(),
                    [] __device__(const thrust::tuple<real, real, real>& vec) {
                      return vec.get<0>() * vec.get<0>() +
                             vec.get<1>() * vec.get<1>() +
                             vec.get<2>() * vec.get<2>();
                    });
}

void find_diagonal_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_row_offsets,
  cusp::array1d<int, cusp::device_memory>::const_view di_row_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_column_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals)
{
  // Iterates over the matrix entry coordinates, and returns whether the row is
  // less than the column, which would mean this is in the upper triangle.
  const auto cmp_less_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(di_row_indices.begin(), di_column_indices.begin())),
    [] __device__(const thrust::tuple<int, int>& coord) {
      return coord.get<0>() < coord.get<1>();
    });
  // Then reduce using the keys to find how many in each column are before
  // the diagonal entry
  thrust::reduce_by_key(di_row_indices.begin(),
                        di_row_indices.end(),
                        cmp_less_it,
                        thrust::make_discard_iterator(),
                        do_diagonals.begin());
  // Sum in the cumulative valence and a count to finalize the diagonal indices
  cusp::blas::axpbypcz(do_diagonals,
                       di_row_offsets,
                       cusp::counting_array<int>(0, do_diagonals.size()),
                       do_diagonals,
                       1,
                       1,
                       1);
}

void make_skip_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_skip_keys,
  cusp::array1d<int, cusp::device_memory>::view do_iterator_indices)
{
  // Start with ones
  cusp::blas::fill(do_iterator_indices, 0);
  // Scatter ones into the indices array
  thrust::copy_n(thrust::constant_iterator<int>(1),
                 di_skip_keys.size(),
                 thrust::make_permutation_iterator(do_iterator_indices.begin(),
                                                   di_skip_keys.begin()));
  // Add the original entry indices to our diagonal scattered ones
  thrust::transform(di_skip_keys.begin(),
                    di_skip_keys.end(),
                    thrust::counting_iterator<int>(0),
                    do_iterator_indices.begin(),
                    thrust::plus<int>());
}

FLO_API void cotangent_laplacian(
  cusp::array1d<real, cusp::device_memory>::const_view di_edges,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian)
{
  // Here we calculate the diagonal indices, and create a diagonal iterator that
  // writes to those, we also create a diagonal skip iterator that avoids them
  find_diagonal_indices(
    di_cumulative_valence, di_adjacency_keys, di_adjacency, do_diagonals);
  // This will be used to permute the value iterator
  auto& diagonal_stride = do_cotangent_laplacian.column_indices;
  make_skip_indices(do_diagonals, diagonal_stride);
  // Iterator that skips the diagonals
  const auto value_it = thrust::make_permutation_iterator(
    do_cotangent_laplacian.values.begin(), diagonal_stride);
  // An iterator for each row, column pair of indices
  auto entry_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_cotangent_laplacian.row_indices.begin(),
                       do_cotangent_laplacian.column_indices.begin()));
  // Iterator for non-diagonal matrix entries
  auto non_diag_begin =
    thrust::make_permutation_iterator(entry_it, diagonal_stride.begin());
  // Iterator for diagonal matrix entries
  auto diag_begin =
    thrust::make_permutation_iterator(entry_it, do_diagonals.begin());


  // Copy the adjacency keys and the adjacency info as the matrix coords
  thrust::copy_n(thrust::make_zip_iterator(thrust::make_tuple(
                   di_adjacency_keys.begin(), di_adjacency.begin())),
                 do_cotangent_laplacian.num_entries,
                 non_diag_begin);

  // Generate the diagonal entry, row and column indices
  thrust::tabulate(
    diag_begin, diag_begin + do_diagonals.size(), [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });

  // Sum all column values into the diagonal value entries
  thrust::reduce_by_key(
    do_cotangent_laplacian.row_indices.begin(),
    do_cotangent_laplacian.row_indices.end(),
    thrust::make_transform_iterator(do_cotangent_laplacian.values.begin(),
                                    thrust::negate<flo::real>()),
    thrust::make_discard_iterator(),
    thrust::make_permutation_iterator(do_cotangent_laplacian.values.begin(),
                                      do_diagonals.begin()));
}

FLO_DEVICE_NAMESPACE_END

