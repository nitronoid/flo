#include "flo/device/cotangent_laplacian_atomic.cuh"
#include "flo/device/thread_util.cuh"
#include "flo/device/adjacency_matrix_indices.cuh"
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__device__ float fast_inv_sqrt(float x) noexcept
{
  const float x2 = x * 0.5f;
  x = __uint_as_float(0x5f3759df - (__float_as_uint(x) >> 1));
  x *= (1.5f - (x2 * x * x));
  return x;
}

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

enum MASK { FULL_MASK = 0xffffffff };
__global__ void
d_cotangent_laplacian_values(const real* __restrict__ di_vertices,
                             const int* __restrict__ di_faces,
                             const int* __restrict__ di_entry,
                             const int32_t i_nverts,
                             const int32_t i_nfaces,
                             real* __restrict__ do_values)
{
  // Calculate our global thread index
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // The face index is the thread index / 4
  const int32_t fid = tid >> 2;
  // Calculate which lane our thread is in: [0,1,2,3]
  const int8_t lane = tid - fid * 4;

  // Guard against threads that would read/write out of bounds
  if (fid >= i_nfaces)
    return;

  // Load vertex points into registers
  real edge_x, edge_y, edge_z;
  // First three threads read from global memory
  if (lane < 3)
  {
    const int32_t pid = di_faces[i_nfaces * lane + fid];
    edge_x = di_vertices[0 * i_nverts + pid];
    edge_y = di_vertices[1 * i_nverts + pid];
    edge_z = di_vertices[2 * i_nverts + pid];
  }
  // Convert our 0,1,2 reads into the 1,2,0,1 layout over four threads
  {
    const int8_t source_lane = (lane + 1) - 3 * (lane > 1);
    edge_x = __shfl_sync(FULL_MASK, edge_x, source_lane, 4);
    edge_y = __shfl_sync(FULL_MASK, edge_y, source_lane, 4);
    edge_z = __shfl_sync(FULL_MASK, edge_z, source_lane, 4);
  }
  // Compute edge vectors from neighbor threads
  // 1-2, 2-0, 0-1, 1-2
  {
    const int8_t source_lane = (lane + 1) - 3 * (lane == 3);
    edge_x -= __shfl_sync(FULL_MASK, edge_x, source_lane, 4);
    edge_y -= __shfl_sync(FULL_MASK, edge_y, source_lane, 4);
    edge_z -= __shfl_sync(FULL_MASK, edge_z, source_lane, 4);
  }

  // Get the components of the neighboring edge
  const real b_x = __shfl_down_sync(FULL_MASK, edge_x, 1, 4);
  const real b_y = __shfl_down_sync(FULL_MASK, edge_y, 1, 4);
  const real b_z = __shfl_down_sync(FULL_MASK, edge_z, 1, 4);

  // Compute the inverse area (1/4A == 1/(4*0.5*x^1/2) == 0.5 * 1/(x^1/2))
  const real inv_area = 0.5f * __frsqrt_rn(sqr(edge_y * b_z - edge_z * b_y) +
                                           sqr(edge_z * b_x - edge_x * b_z) +
                                           sqr(edge_x * b_y - edge_y * b_x));

  // Dot product with neighbor
  edge_x = (edge_x * b_x + edge_y * b_y + edge_z * b_z) * inv_area;

  if (lane < 3)
  {
    const int32_t address_lower = di_entry[i_nfaces * lane + fid];
    const int32_t address_upper = di_entry[i_nfaces * (lane + 3) + fid];

    // Write the row and column indices
    atomicAdd(do_values + address_lower, edge_x);
    atomicAdd(do_values + address_upper, edge_x);
  }
}

}  // namespace

FLO_API void cotangent_laplacian_values(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian)
{
  const size_t block_width = 1024;
  const size_t nblocks = di_faces.num_cols * 4 / block_width + 1;
  // const dim3 block_dim{341, 3, 1};
  // const size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  // const size_t nblocks = di_faces.num_cols * 3 / nthreads_per_block + 1;
  // const size_t shared_memory_size = sizeof(flo::real) * block_dim.x * 15;
  d_cotangent_laplacian_values<<<nblocks, block_width>>>(
    di_vertices.values.begin().base().get(),
    di_faces.values.begin().base().get(),
    di_entry_indices.values.begin().base().get(),
    di_vertices.num_cols,
    di_faces.num_cols,
    do_cotangent_laplacian.values.begin().base().get());
  cudaDeviceSynchronize();
  // Reduce by row to produce diagonal values
  thrust::reduce_by_key(
    do_cotangent_laplacian.row_indices.begin(),
    do_cotangent_laplacian.row_indices.end(),
    do_cotangent_laplacian.values.begin(),
    thrust::make_discard_iterator(),
    thrust::make_transform_output_iterator(
      thrust::make_permutation_iterator(do_cotangent_laplacian.values.begin(),
                                        di_diagonals.begin()),
      thrust::negate<real>()));
}

FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian)
{
  auto temp_ptr = thrust::device_pointer_cast(
      reinterpret_cast<void*>(do_cotangent_laplacian.values.begin().base().get()));
  adjacency_matrix_indices(di_faces,
                           di_adjacency_keys,
                           di_adjacency,
                           di_cumulative_valence,
                           do_entry_indices,
                           do_diagonal_indices,
                           do_cotangent_laplacian.row_indices,
                           do_cotangent_laplacian.column_indices,
                           temp_ptr);

  cusp::array2d<int, cusp::device_memory>::const_view const_entry_indices(
      do_entry_indices.num_rows,
      do_entry_indices.num_cols,
      1,
      do_entry_indices.values);

  cotangent_laplacian_values(di_vertices,
                             di_faces,
                             const_entry_indices,
                             do_diagonal_indices,
                             do_cotangent_laplacian);

}

FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian)
{
  cusp::array2d<int, cusp::device_memory> entry_indices(6, di_faces.num_cols);
  cusp::array1d<int, cusp::device_memory> diagonal_indices(di_vertices.num_cols);

  auto temp_ptr = thrust::device_pointer_cast(
      reinterpret_cast<void*>(do_cotangent_laplacian.values.begin().base().get()));
  adjacency_matrix_indices(di_faces,
                           di_adjacency_keys,
                           di_adjacency,
                           di_cumulative_valence,
                           entry_indices,
                           diagonal_indices,
                           do_cotangent_laplacian.row_indices,
                           do_cotangent_laplacian.column_indices,
                           temp_ptr);

  cotangent_laplacian_values(di_vertices,
                             di_faces,
                             entry_indices,
                             diagonal_indices,
                             do_cotangent_laplacian);

}

FLO_DEVICE_NAMESPACE_END

