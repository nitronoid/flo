#include "flo/device/adjacency_matrix_indices.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>
#include <cusp/blas/blas.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void
d_adjacency_matrix_indices(const int* __restrict__ di_faces,
                           const int* __restrict__ di_vertex_adjacency,
                           const int* __restrict__ di_cumulative_valence,
                           const int i_nfaces,
                           int* __restrict__ do_indices)
{
  const int fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Determine whether we are calculating a column or row major offset
  // even threads are col major while odd ones are row major
  const uint8_t major = threadIdx.y >= 3;

  const uchar3 loop = tri_edge_loop(threadIdx.y - 3 * major);

  // Global vertex indices that make this edge
  const int2 edge =
    make_int2(di_faces[i_nfaces * nth_element(loop, major) + fid],
              di_faces[i_nfaces * nth_element(loop, !major) + fid]);

  int begin = di_cumulative_valence[edge.x];
  int end = di_cumulative_valence[edge.x + 1] - 1;

  auto iter = thrust::lower_bound(thrust::seq,
                                  di_vertex_adjacency + begin,
                                  di_vertex_adjacency + end,
                                  edge.y);
  const int index = (iter - di_vertex_adjacency) + edge.x + (edge.y > edge.x);
  do_indices[i_nfaces * threadIdx.y + fid] = index;
}

}  // namespace

FLO_API void adjacency_matrix_indices(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::array1d<int, cusp::device_memory>::view do_row_indices,
  cusp::array1d<int, cusp::device_memory>::view do_column_indices,
  thrust::device_ptr<void> dio_temp)
{
  // Find the diagonal matrix entry indices
  find_diagonal_indices(
    di_cumulative_valence, di_adjacency_keys, di_adjacency, do_diagonal_indices);

  const int ndiagonals = do_diagonal_indices.size();
  const int nnon_diagonals = do_column_indices.size() - ndiagonals;

  // This will be used to permute the value iterator
  thrust::device_ptr<int> diagonal_stride_ptr{
    reinterpret_cast<int*>(dio_temp.get())};
  auto diagonal_stride = cusp::make_array1d_view(
    diagonal_stride_ptr, diagonal_stride_ptr + nnon_diagonals);

  make_skip_indices(do_diagonal_indices, diagonal_stride);
  // An iterator for each row, column pair of indices
  auto entry_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_row_indices.begin(),
                       do_column_indices.begin()));
  // Iterator for non-diagonal matrix entries
  auto non_diag_begin =
    thrust::make_permutation_iterator(entry_it, diagonal_stride.begin());
  // Copy the adjacency keys and the adjacency info as the matrix coords
  thrust::copy_n(thrust::make_zip_iterator(thrust::make_tuple(
                   di_adjacency_keys.begin(), di_adjacency.begin())),
                 nnon_diagonals,
                 non_diag_begin);
  // Iterator for diagonal matrix entries
  auto diag_begin =
    thrust::make_permutation_iterator(entry_it, do_diagonal_indices.begin());
  // Generate the diagonal entry, row and column indices
  thrust::tabulate(
    diag_begin, diag_begin + do_diagonal_indices.size(), 
    [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });


  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  const int nblocks =
    di_faces.num_cols * 6 / (block_dim.x * block_dim.y * block_dim.z) + 1;

  d_adjacency_matrix_indices<<<nblocks, block_dim>>>(
    di_faces.values.begin().base().get(),
    di_adjacency.begin().base().get(),
    di_cumulative_valence.begin().base().get(),
    di_faces.num_cols,
    do_entry_indices.values.begin().base().get());
  cudaDeviceSynchronize();
}

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
  thrust::fill(do_iterator_indices.begin(), do_iterator_indices.end(), 0);
  // Add ones in locations where a diagonal exists, need atomic due to neighbours
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


