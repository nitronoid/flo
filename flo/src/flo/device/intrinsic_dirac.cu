#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/thread_util.cuh"
#include "flo/device/matrix_operation.cuh"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
template <typename T>
__device__ __forceinline__ constexpr T sqr(T&& i_value) noexcept
{
  return i_value * i_value;
}
__device__ real4 hammilton_product(const real3& i_rhs, const real3& i_lhs)
{
  const real a1 = 0.f;
  const real b1 = i_rhs.x;
  const real c1 = i_rhs.y;
  const real d1 = i_rhs.z;
  const real a2 = 0.f;
  const real b2 = i_lhs.x;
  const real c2 = i_lhs.y;
  const real d2 = i_lhs.z;
  // W is last in a vector
  return make_float4(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
                     a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2);
}
__device__ real4 hammilton_product(
  real lhs_x, real lhs_y, real lhs_z, real rhs_x, real rhs_y, real rhs_z)
{
  const real a1 = 0.f;
  const real b1 = lhs_x;
  const real c1 = lhs_y;
  const real d1 = lhs_z;
  const real a2 = 0.f;
  const real b2 = rhs_x;
  const real c2 = rhs_y;
  const real d2 = rhs_z;
  // W is last in a vector
  return make_float4(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
                     a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2);
}

template <typename T>
__device__ constexpr T reciprocal(T&& i_value) noexcept
{
  return T{1} / i_value;
}

__device__ real4 make_quat(real4 i_value) noexcept
{
  return i_value;
}

__device__ real4 make_quat(real i_value) noexcept
{
  real4 q;
  q.x = 0.f;
  q.y = 0.f;
  q.z = 0.f;
  q.w = i_value;
  return q;
}

template <
  typename T,
  typename = typename std::enable_if<std::is_same<T, real>::value ||
                                     std::is_same<T, real4>::value>::type>
__global__ void
d_to_real_quaternion_matrix(const int* __restrict__ di_rows,
                            const int* __restrict__ di_columns,
                            const T* __restrict__ di_values,
                            const int* __restrict__ di_cumulative_column_size,
                            const int i_nvalues,
                            int* __restrict__ do_rows,
                            int* __restrict__ do_columns,
                            real* __restrict__ do_values)
{
  // Declare one shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Offset our shared memory pointer by the number of values * sizeof(real4)
  int32_t* __restrict__ row_index = (int32_t*)(shared_memory);
  // Offset our shared memory pointer by the number of values * sizeof(int)
  int32_t* __restrict__ col_index = (int32_t*)(row_index + blockDim.x * 4);
  // Create pointers into the block dividing it for the different uses
  real4* __restrict__ quaternion_entry = (real4*)(col_index + blockDim.x * 4);

  // Calculate which entry this thread is transforming
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Guard against out of range threads
  if (global_id >= i_nvalues)
    return;

  // Get our block local id for shared memory access
  const int16_t local_id = threadIdx.x * 4 + threadIdx.y;

  if (!threadIdx.y)
  {
    // Read the quaternion entry once
    const real4 quat = make_quat(di_values[global_id]);
    // Copy the quaternion across shared memory so all threads have access
    quaternion_entry[threadIdx.x * 4 + 0] = quat;
    quaternion_entry[threadIdx.x * 4 + 1] = quat;
    quaternion_entry[threadIdx.x * 4 + 2] = quat;
    quaternion_entry[threadIdx.x * 4 + 3] = quat;
    // Read the row index once
    const int row = di_rows[global_id];
    // Copy across shared memory so all threads have access
    row_index[threadIdx.x * 4 + 0] = row;
    row_index[threadIdx.x * 4 + 1] = row;
    row_index[threadIdx.x * 4 + 2] = row;
    row_index[threadIdx.x * 4 + 3] = row;
    // Read the column index once
    const int col = di_columns[global_id];
    // Copy across shared memory so all threads have access
    col_index[threadIdx.x * 4 + 0] = col;
    col_index[threadIdx.x * 4 + 1] = col;
    col_index[threadIdx.x * 4 + 2] = col;
    col_index[threadIdx.x * 4 + 3] = col;
  }
  __syncthreads();
  const uint8_t sign = (0x284E >> (threadIdx.y * 4u)) & 15u;
  const uchar4 loop = quat_loop(threadIdx.y);
  real4 quat;
  quat.x =
    nth_element(quaternion_entry[local_id], loop.x) * sign_from_bit(sign, 0u);
  quat.y =
    nth_element(quaternion_entry[local_id], loop.y) * sign_from_bit(sign, 1u);
  quat.z =
    nth_element(quaternion_entry[local_id], loop.z) * sign_from_bit(sign, 2u);
  quat.w =
    nth_element(quaternion_entry[local_id], loop.w) * sign_from_bit(sign, 3u);

  // Calculate where we're writing to
  const int32_t prev_col_offset =
    di_cumulative_column_size[row_index[local_id] - 1];
  const int32_t curr_col_offset =
    di_cumulative_column_size[row_index[local_id]] - prev_col_offset;
  const int32_t offset = prev_col_offset * 16 +
                         curr_col_offset * 4 * threadIdx.y +
                         (global_id - prev_col_offset) * 4;

  // Use a vector cast to write using a 16 byte instruction
  *reinterpret_cast<real4*>(do_values + offset) = quat;
  int32_t C = col_index[local_id] * 4;
  *reinterpret_cast<int4*>(do_columns + offset) =
    make_int4(C, C + 1, C + 2, C + 3);
  *reinterpret_cast<int4*>(do_rows + offset) =
    make_int4(row_index[local_id] * 4 + threadIdx.y);
}

enum MASK { FULL_MASK = 0xffffffff };
// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void d_intrinsic_dirac_atomic(const real* __restrict__ di_vertices,
                                         const int* __restrict__ di_faces,
                                         const real* __restrict__ di_rho,
                                         const int* __restrict__ di_entry,
                                         const int i_nverts,
                                         const int i_nfaces,
                                         real4* __restrict__ do_values)
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
  real edge_x, edge_y, edge_z, rho;
  // First three threads read from global memory
  if (lane < 3)
  {
    const int32_t pid = di_faces[i_nfaces * lane + fid];
    edge_x = di_vertices[0 * i_nverts + pid];
    edge_y = di_vertices[1 * i_nverts + pid];
    edge_z = di_vertices[2 * i_nverts + pid];
    rho = di_rho[pid];
  }
  // Convert our 0,1,2 reads into the 1,2,0,1 layout over four threads
  {
    const int8_t source_lane = (lane + 1) - 3 * (lane > 1);
    edge_x = __shfl_sync(FULL_MASK, edge_x, source_lane, 4);
    edge_y = __shfl_sync(FULL_MASK, edge_y, source_lane, 4);
    edge_z = __shfl_sync(FULL_MASK, edge_z, source_lane, 4);
  }
  {
    const int8_t source_lane = lane - 3 * (lane == 3);
    rho = __shfl_sync(FULL_MASK, rho, source_lane, 4);
  }
  // Compute edge vectors from neighbor threads
  // 1-2, 2-0, 0-1, 1-2
  // 2-1, 0-2, 1-0, 2-1
  {
    const int8_t source_lane = (lane + 1) - 3 * (lane == 3);
    edge_x = __shfl_sync(FULL_MASK, edge_x, source_lane, 4) - edge_x;
    edge_y = __shfl_sync(FULL_MASK, edge_y, source_lane, 4) - edge_y;
    edge_z = __shfl_sync(FULL_MASK, edge_z, source_lane, 4) - edge_z;
  }

  // Get the components of the neighboring edge
  const real b_x = __shfl_down_sync(FULL_MASK, edge_x, 1, 4);
  const real b_y = __shfl_down_sync(FULL_MASK, edge_y, 1, 4);
  const real b_z = __shfl_down_sync(FULL_MASK, edge_z, 1, 4);
  const real b_rho = __shfl_down_sync(FULL_MASK, rho, 1, 4);

  // Compute the inverse area (1/-4A == 1/(-4*0.5*x^1/2) == -0.5 * 1/(x^1/2))
  const real inv_area = -0.5f * __frsqrt_rn(sqr(edge_y * b_z - edge_z * b_y) +
                                            sqr(edge_z * b_x - edge_x * b_z) +
                                            sqr(edge_x * b_y - edge_y * b_x));

  const real c = ((1.f / inv_area) * -0.25f) * reciprocal(9.f) * rho * b_rho;
  const real4 img = make_float4(reciprocal(6.f) * (rho * b_x - b_rho * edge_x),
                                reciprocal(6.f) * (rho * b_y - b_rho * edge_y),
                                reciprocal(6.f) * (rho * b_z - b_rho * edge_z),
                                0.f);

  // Compute lower result TODO: Vectorized atomics
  if (lane < 3)
  {
    real4 result =
      hammilton_product(edge_x, edge_y, edge_z, b_x, b_y, b_z) * inv_area + img;
    result.w += c;

    const int32_t address = di_entry[i_nfaces * lane + fid];

    auto out = reinterpret_cast<real*>(do_values + address);
    atomicAdd(out + 0, result.x);
    atomicAdd(out + 1, result.y);
    atomicAdd(out + 2, result.z);
    atomicAdd(out + 3, result.w);
  }

  // Compute upper result
  if (lane < 3)
  {
    real4 result =
      hammilton_product(b_x, b_y, b_z, edge_x, edge_y, edge_z) * inv_area - img;
    result.w += c;

    const int32_t address = di_entry[i_nfaces * (lane + 3) + fid];

    auto out = reinterpret_cast<real*>(do_values + address);
    atomicAdd(out + 0, result.x);
    atomicAdd(out + 1, result.y);
    atomicAdd(out + 2, result.z);
    atomicAdd(out + 3, result.w);
  }
}

struct dirac_diagonal
  : public thrust::unary_function<thrust::tuple<int, const int>, flo::real4>
{
  dirac_diagonal(thrust::device_ptr<const real> di_vertices,
                 thrust::device_ptr<const int> di_faces,
                 thrust::device_ptr<const real> di_face_area,
                 thrust::device_ptr<const real> di_rho,
                 int32_t i_nverts,
                 int32_t i_nfaces)
    : di_vertices(std::move(di_vertices.get()))
    , di_faces(std::move(di_faces.get()))
    , di_face_area(std::move(di_face_area.get()))
    , di_rho(std::move(di_rho.get()))
    , nverts(std::move(i_nverts))
    , nfaces(std::move(i_nfaces))
  {
  }

  const real* __restrict__ di_vertices;
  const int* __restrict__ di_faces;
  const real* __restrict__ di_face_area;
  const real* __restrict__ di_rho;
  const int32_t nverts;
  const int32_t nfaces;

  __host__ __device__ flo::real4
  operator()(thrust::tuple<int, const int> id) const
  {
    const int vid = id.get<0>();
    const int fid = id.get<1>();
    // Remove vid from faces[fid]
    int2 e_id;
    {
      const int fx = di_faces[fid + 0 * nfaces];
      const int fy = di_faces[fid + 1 * nfaces];
      const int fz = di_faces[fid + 2 * nfaces];
      if (fx == vid)
        e_id = make_int2(fy, fz);
      if (fy == vid)
        e_id = make_int2(fx, fz);
      if (fz == vid)
        e_id = make_int2(fx, fy);
    }

    const real ex =
      di_vertices[e_id.y + 0 * nverts] - di_vertices[e_id.x + 0 * nverts];
    const real ey =
      di_vertices[e_id.y + 1 * nverts] - di_vertices[e_id.x + 1 * nverts];
    const real ez =
      di_vertices[e_id.y + 2 * nverts] - di_vertices[e_id.x + 2 * nverts];

    const real rho = di_rho[vid];
    const real area = di_face_area[fid];

    flo::real4 o_val;
    o_val.x = o_val.y = o_val.z = 0.f;
    o_val.w = (ex * ex + ey * ey + ez * ez) / (4.f * area) +
              (rho * rho * area) * reciprocal(9.f);
    return o_val;
  }
};

template <
  typename T,
  typename = typename std::enable_if<std::is_same<T, real>::value ||
                                     std::is_same<T, real4>::value>::type>
void to_real_quaternion_matrix_impl(
  typename cusp::coo_matrix<int, T, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix)
{
  dim3 block_dim;
  block_dim.z = 1;
  block_dim.y = 4;
  block_dim.x = 256;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks =
    di_quaternion_matrix.values.size() * 4 / nthreads_per_block + 1;
  size_t shared_memory_size =
    (sizeof(flo::real4) + sizeof(int32_t) * 2) * nthreads_per_block;

  d_to_real_quaternion_matrix<<<nblocks, block_dim, shared_memory_size>>>(
    di_quaternion_matrix.row_indices.begin().base().get(),
    di_quaternion_matrix.column_indices.begin().base().get(),
    di_quaternion_matrix.values.begin().base().get(),
    di_cumulative_column_size.begin().base().get(),
    di_quaternion_matrix.values.size(),
    do_real_matrix.row_indices.begin().base().get(),
    do_real_matrix.column_indices.begin().base().get(),
    do_real_matrix.values.begin().base().get());
  cudaDeviceSynchronize();
}

}  // namespace

FLO_API void to_quaternion_matrix(
  cusp::coo_matrix<int, real4, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix)
{
  to_real_quaternion_matrix_impl<real4>(
    di_quaternion_matrix, di_cumulative_column_size, do_real_matrix);
}

FLO_API void to_real_quaternion_matrix(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_quaternion_matrix,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_column_size,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_real_matrix)
{
  to_real_quaternion_matrix_impl<real>(
    di_quaternion_matrix, di_cumulative_column_size, do_real_matrix);
}

FLO_API void intrinsic_dirac(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_offset,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix)
{
  // Find the diagonal matrix entry indices
  find_diagonal_indices(
    di_cumulative_valence, di_adjacency_keys, di_adjacency, do_diagonals);

  const int ndiagonals = do_diagonals.size();
  const int nnon_diagonals = do_dirac_matrix.num_entries - ndiagonals;

  // This will be used to permute the value iterator
  thrust::device_ptr<int> diagonal_stride_ptr{
    reinterpret_cast<int*>(do_dirac_matrix.values.begin().base().get())};
  auto diagonal_stride = cusp::make_array1d_view(
    diagonal_stride_ptr, diagonal_stride_ptr + nnon_diagonals);

  make_skip_indices(do_diagonals, diagonal_stride);
  // An iterator for each row, column pair of indices
  auto entry_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_dirac_matrix.row_indices.begin(),
                       do_dirac_matrix.column_indices.begin()));
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
  // When passing the face and offset data to cuda, we reinterpret them as int
  // arrays. The advantage of this is coalesced memory reads by neighboring
  // threads, and access at a more granular level.
  // The cast is inherently safe due to the alignment of cuda vector types,
  // and reinterpret casting guarantees no changes to the underlying values
  d_intrinsic_dirac_atomic<<<nblocks, block_width>>>(
    di_vertices.values.begin().base().get(),
    di_faces.values.begin().base().get(),
    di_rho.begin().base().get(),
    di_entry_offset.values.begin().base().get(),
    di_vertices.num_cols,
    di_faces.num_cols,
    do_dirac_matrix.values.begin().base().get());
  cudaDeviceSynchronize();

  // Iterate over adjacent faces and the corresponding vertex id
  auto face_vertex_iter = thrust::make_zip_iterator(
    thrust::make_tuple(di_vertex_triangle_adjacency_keys.begin(),
                       di_vertex_triangle_adjacency.begin()));

  // Transform opposing edge's, found through the vertex triangle adjacency
  // information, into diagonal dirac contributions
  // Doing this through the iterator saves a memory allocation
  auto dirac_iter = thrust::make_transform_iterator(
    face_vertex_iter,
    dirac_diagonal(di_vertices.values.begin().base(),
                   di_faces.values.begin().base(),
                   di_face_area.begin().base(),
                   di_rho.begin().base(),
                   di_vertices.num_cols,
                   di_faces.num_cols));

  thrust::reduce_by_key(
    di_vertex_triangle_adjacency_keys.begin(),
    di_vertex_triangle_adjacency_keys.end(),
    dirac_iter,
    thrust::make_discard_iterator(),
    thrust::make_permutation_iterator(do_dirac_matrix.values.begin(),
                                      do_diagonals.begin()));
}

FLO_DEVICE_NAMESPACE_END

