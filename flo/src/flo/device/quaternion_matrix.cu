#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
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

FLO_DEVICE_NAMESPACE_END


