#include "flo/device/cotangent_laplacian_atomic.cuh"
#include "flo/device/thread_util.cuh"
//#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <cusp/print.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
#ifdef FLO_USE_DOUBLE_PRECISION
__device__ double atomicAdd(double* __restrict__ i_address, const double i_val)
{
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(i_address);
  unsigned long long int old = *address_as_ull, ass;
  do
  {
    ass = old;
    old = atomicCAS(address_as_ull,
                    ass,
                    __double_as_longlong(i_val + __longlong_as_double(ass)));
  } while (ass != old);
  return __longlong_as_double(old);
}
#endif

template <typename T>
__device__ __forceinline__ constexpr T sqr(T&& i_value) noexcept
{
  return i_value * i_value;
}
// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void
d_cotangent_laplacian_values(const real* __restrict__ di_vertices,
                             const int* __restrict__ di_faces,
                             const real* __restrict__ di_face_area,
                             const int* __restrict__ di_entry,
                             const int i_nverts,
                             const int i_nfaces,
                             real* __restrict__ do_values)
{
  // Declare one shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Create pointers into the block dividing it for the different uses
  // Area is duplicated for each thread, 6 per face
  real* __restrict__ area = (real*)shared_memory;
  // Each thread produces a resulting value, 6 per face
  real* __restrict__ result = (real*)(area + blockDim.x * 3);
  // Each thread reads an X,Y,Z point, 6 per face
  real* __restrict__ points_x = (real*)(result + blockDim.x * 3);
  real* __restrict__ points_y = (real*)(points_x + blockDim.x * 3);
  real* __restrict__ points_z = (real*)(points_y + blockDim.x * 3);

  // Calculate which face this thread is acting on
  const int32_t fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Only write once per face
  if (!threadIdx.y)
  {
    // Duplicate for each thread to reduce bank conflicts
    const real face_area = 1.f / (di_face_area[fid] * 4.f);
    area[blockDim.x * 0 + threadIdx.x] = face_area;
    area[blockDim.x * 1 + threadIdx.x] = face_area;
    area[blockDim.x * 2 + threadIdx.x] = face_area;
  }

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uchar3 loop = tri_edge_loop(threadIdx.y);
  // Compute local indices rotated by the corner this thread corresponds to
  const int16_t v0 = blockDim.x * loop.x + threadIdx.x;
  const int16_t v1 = blockDim.x * loop.y + threadIdx.x;
  const int16_t v2 = blockDim.x * loop.z + threadIdx.x;

  // Write the vertex positions into shared memory
  // We offset using nverts and the threadIdx Y value to pick which column of
  // the point and face matrices to read from.
  {
    const int32_t pid = di_faces[i_nfaces * threadIdx.y + fid];
    points_x[v0] = di_vertices[0 * i_nverts + pid];
    points_y[v0] = di_vertices[1 * i_nverts + pid];
    points_z[v0] = di_vertices[2 * i_nverts + pid];
  }
  __syncthreads();

  // Compute the final result, (e0*e1) / (area*4)
  result[v0] = ((points_x[v1] - points_x[v2]) * (points_x[v0] - points_x[v2]) +
                (points_y[v1] - points_y[v2]) * (points_y[v0] - points_y[v2]) +
                (points_z[v1] - points_z[v2]) * (points_z[v0] - points_z[v2])) *
               area[v0];

  const int32_t address_lower = di_entry[i_nfaces * threadIdx.y + fid];
  const int32_t address_upper = di_entry[i_nfaces * (threadIdx.y + 3) + fid];

  // Write the row and column indices
  atomicAdd(do_values + address_lower, -result[v0]);
  atomicAdd(do_values + address_upper, -result[v0]);
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

void make_skip_indices(
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

}  // namespace

FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_offset,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian)
{
  // Find the diagonal matrix entry indices
  find_diagonal_indices(
    di_cumulative_valence, di_adjacency_keys, di_adjacency, do_diagonals);

  const int ndiagonals = do_diagonals.size();
  const int nnon_diagonals = do_cotangent_laplacian.num_entries - ndiagonals;

  // This will be used to permute the value iterator
  thrust::device_ptr<int> diagonal_stride_ptr{
    reinterpret_cast<int*>(do_cotangent_laplacian.values.begin().base().get())};
  auto diagonal_stride = cusp::make_array1d_view(
    diagonal_stride_ptr, diagonal_stride_ptr + nnon_diagonals);

  make_skip_indices(do_diagonals, diagonal_stride);
  // An iterator for each row, column pair of indices
  auto entry_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_cotangent_laplacian.row_indices.begin(),
                       do_cotangent_laplacian.column_indices.begin()));
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
    thrust::make_permutation_iterator(entry_it, do_diagonals.begin());
  // Generate the diagonal entry, row and column indices
  thrust::tabulate(
    diag_begin, diag_begin + do_diagonals.size(), [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });

  dim3 block_dim;
  block_dim.y = 3;
  block_dim.x = 341;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = di_faces.num_cols * 3 / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size = sizeof(flo::real) * block_dim.x * 15;
  d_cotangent_laplacian_values<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices.values.begin().base().get(),
    di_faces.values.begin().base().get(),
    di_face_area.begin().base().get(),
    di_entry_offset.values.begin().base().get(),
    di_vertices.num_cols,
    di_faces.num_cols,
    do_cotangent_laplacian.values.begin().base().get());
  cudaDeviceSynchronize();
  // Reduce by row to produce diagonal values
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

