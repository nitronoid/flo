#include "flo/device/cotangent_laplacian_atomic.cuh"
#include "flo/device/thread_util.cuh"
#include "flo/device/matrix_operation.cuh"
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

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
  // Area is duplicated for each thread, 3 per face
  real* __restrict__ area = (real*)shared_memory;
  // Each thread produces a resulting value, 3 per face
  real* __restrict__ result = (real*)(area + blockDim.x * 3);
  // Each thread reads an X,Y,Z point, 3 per face
  real* __restrict__ points_x = (real*)(result + blockDim.x * 3);
  real* __restrict__ points_y = (real*)(points_x + blockDim.x * 3);
  real* __restrict__ points_z = (real*)(points_y + blockDim.x * 3);

  // Calculate which face this thread is acting on
  const int32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool in_range = fid < i_nfaces;

  // Need these outside the if-scope
  int16_t v0, v1, v2;
  // Guard against out of bounds memory writing
  if (in_range)
  {
    // Only write once per face
    if (!threadIdx.y)
    {
      // Duplicate for each thread to reduce bank conflicts
      const real face_area = 1.f / (di_face_area[fid] * 4.f);
      area[blockDim.x * 0 + threadIdx.x] = face_area;
      area[blockDim.x * 1 + threadIdx.x] = face_area;
      area[blockDim.x * 2 + threadIdx.x] = face_area;
    }

    // Get the vertex order, need to half the tid as we have two threads per
    // edge
    const uchar3 loop = tri_edge_loop(threadIdx.y);
    // Compute local indices rotated by the corner this thread corresponds to
    v0 = blockDim.x * loop.x + threadIdx.x;
    v1 = blockDim.x * loop.y + threadIdx.x;
    v2 = blockDim.x * loop.z + threadIdx.x;

    // Write the vertex positions into shared memory
    // We offset using nverts and the threadIdx Y value to pick which column of
    // the point and face matrices to read from.
    {
      const int32_t pid = di_faces[i_nfaces * threadIdx.y + fid];
      points_x[v0] = di_vertices[0 * i_nverts + pid];
      points_y[v0] = di_vertices[1 * i_nverts + pid];
      points_z[v0] = di_vertices[2 * i_nverts + pid];
    }
  }
  // Ensure all writes to shared memory are complete
  __syncthreads();
  // Guard against out of bounds memory writing
  if (in_range)
  {
    // Compute the final result, (e0*e1) / (area*4)
    result[v0] =
      ((points_x[v1] - points_x[v2]) * (points_x[v0] - points_x[v2]) +
       (points_y[v1] - points_y[v2]) * (points_y[v0] - points_y[v2]) +
       (points_z[v1] - points_z[v2]) * (points_z[v0] - points_z[v2])) *
      area[v0];

    const int32_t address_lower = di_entry[i_nfaces * threadIdx.y + fid];
    const int32_t address_upper = di_entry[i_nfaces * (threadIdx.y + 3) + fid];

    // Write the row and column indices
    atomicAdd(do_values + address_lower, -result[v0]);
    atomicAdd(do_values + address_upper, -result[v0]);
  }
}

enum MASK { FULL_MASK = 0xffffffff };
__global__ void
d_cotangent_laplacian_valuees(const real* __restrict__ di_vertices,
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
  const real inv_area = 0.5f * rsqrtf(sqr(edge_y * b_z - edge_z * b_y) +
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

  const size_t block_width = 1024;
  const size_t nblocks = di_faces.num_cols * 4 / block_width + 1;
  // const dim3 block_dim{341, 3, 1};
  // const size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  // const size_t nblocks = di_faces.num_cols * 3 / nthreads_per_block + 1;
  // const size_t shared_memory_size = sizeof(flo::real) * block_dim.x * 15;
  d_cotangent_laplacian_valuees<<<nblocks, block_width>>>(
    di_vertices.values.begin().base().get(),
    di_faces.values.begin().base().get(),
    di_entry_offset.values.begin().base().get(),
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
                                        do_diagonals.begin()),
      thrust::negate<real>()));
}

FLO_DEVICE_NAMESPACE_END

