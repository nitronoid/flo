#include "flo/device/matrix_operation.cuh"
#include <cusp/blas/blas.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void find_diagonal_indices(
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
    [] __host__ __device__(const thrust::tuple<int, int>& coord) -> int {
      return coord.get<0>() > coord.get<1>();
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
                       di_row_offsets.subarray(0, do_diagonals.size()),
                       cusp::counting_array<int>(do_diagonals.size()),
                       do_diagonals,
                       1,
                       1,
                       1);
}

FLO_API void make_skip_indices(
  cusp::array1d<int, cusp::device_memory>::const_view di_skip_keys,
  cusp::array1d<int, cusp::device_memory>::view do_iterator_indices)
{
  // Start with zeros
  cusp::blas::fill(do_iterator_indices, 0);
  // Add ones in locations where a diagonal exists
  thrust::for_each_n(
    thrust::counting_iterator<int>(0),
    di_skip_keys.size(),
    [skip_keys = di_skip_keys.begin().base().get(),
     out = do_iterator_indices.begin().base().get()] __device__(int x) {
      const int skip = skip_keys[x] - x;
      atomicAdd(out + skip, 1);
    });
  // Scan the diagonal markers to produce an offset
  thrust::inclusive_scan(do_iterator_indices.begin(),
                         do_iterator_indices.end(),
                         do_iterator_indices.begin());
  // Add the original entry indices to our offset array
  thrust::transform(do_iterator_indices.begin(),
                    do_iterator_indices.end(),
                    thrust::counting_iterator<int>(0),
                    do_iterator_indices.begin(),
                    thrust::plus<int>());
}

FLO_DEVICE_NAMESPACE_END

