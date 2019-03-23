#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
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

template <typename T>
__device__ constexpr T reciprocal(T&& i_value) noexcept
{
  return T{1} / i_value;
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
  const uint global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Guard against out of range threads
  if (global_id >= i_nvalues)
    return;

  // Get our block local id for shared memory access
  const uint16_t local_id = threadIdx.x * 4 + threadIdx.y;

  if (!threadIdx.y)
  {
    // Read the quaternion entry once
    const real4 quat = make_float4(di_values[global_id]);
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
  const uint8_t sign = (0x5390 >> (threadIdx.y * 4u)) & 15u;
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
  const uint32_t prev_col_offset =
    di_cumulative_column_size[row_index[local_id] - 1];
  const uint32_t curr_col_offset =
    di_cumulative_column_size[row_index[local_id]] - prev_col_offset;
  const uint32_t offset = prev_col_offset * 16 +
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

// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void
d_intrinsic_dirac_atomic(const real3* __restrict__ di_vertices,
                         const int* __restrict__ di_faces,
                         const real* __restrict__ di_face_area,
                         const real* __restrict__ di_rho,
                         const int* __restrict__ di_entry_offset,
                         const uint i_nfaces,
                         int* __restrict__ do_rows,
                         int* __restrict__ do_columns,
                         real4* __restrict__ do_values)
{
  // Declare one shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Create pointers into the block dividing it for the different uses
  real* __restrict__ face_area = (real*)shared_memory;
  // There is a cached point for each corner of the face so we offset
  real3* __restrict__ points = (real3*)(face_area + blockDim.x * 3);
  // There is a cached edge for each corner of the face so we offset
  real3* __restrict__ edges = (real3*)(points + blockDim.x * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* __restrict__ edge_norm2 = (real*)(edges + blockDim.x * 3);
  // There are nfaces *3 squared edge lengths (duplicated for each face vertex)
  real* __restrict__ rho = (real*)(edge_norm2 + blockDim.x * 3);
  // There are nfaces *3 vertex rho values (duplicated for each face vertex)
  uint32_t* __restrict__ eid = (uint32_t*)(rho + blockDim.x * 3);

  // Calculate which face this thread is acting on
  const uint fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uchar3 loop = tri_edge_loop(threadIdx.y >> 1);

  // Compute local edge indices rotated by the corner this thread corresponds to
  const uint16_t local_e0 = threadIdx.x * 3 + loop.x;
  const uint16_t local_e1 = threadIdx.x * 3 + loop.y;
  const uint16_t local_e2 = threadIdx.x * 3 + loop.z;

  // This thread will write to column or row major triangle based on even or odd
  const uint8_t major = !(threadIdx.y & 1);

  // Only write once per face
  if (!threadIdx.y)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    face_area[local_e0] = face_area[local_e1] = face_area[local_e2] =
      di_face_area[fid];
  }
  // Write the vertex positions into shared memory
  if (major)
  {
    const uint32_t vid = di_faces[fid * 3 + loop.x];
    rho[local_e0] = di_rho[vid];
    points[local_e0] = di_vertices[vid];
  }
  __syncthreads();
  // Compute squared length of edges and write to shared memory
  if (major)
  {
    edges[local_e0] = points[local_e2] - points[local_e1];
  }
  __syncthreads();
  const uint16_t O1 = threadIdx.x * 3 + nth_element(loop, 1 + major);
  const uint16_t O2 = threadIdx.x * 3 + nth_element(loop, 1 + !major);
  // Calc imaginary part
  real4 value =
    hammilton_product(edges[O1], edges[O2]) / (-4.f * face_area[local_e0]);
  value +=
    make_float4(reciprocal(6.f) * (rho[O1] * edges[O2] - rho[O2] * edges[O1]));
  // Add real part
  value.w += rho[O1] * rho[O2] * (face_area[local_e0] * reciprocal(9.f));

  // Write the opposing edge ID's into shared memory to reduce global reads
  eid[local_e0 * 2 + !major] =
    di_faces[fid * 3 + nth_element(loop, 1 + !major)];
  __syncthreads();

  const uint32_t R = eid[local_e0 * 2 + !major];
  const uint32_t C = eid[local_e0 * 2 + major];
  const uint32_t address = di_entry_offset[fid * 6 + threadIdx.y] + R + (C > R);
  // Write the row and column indices
  do_rows[address] = R;
  do_columns[address] = C;
  auto out = reinterpret_cast<real*>(do_values + address);
  atomicAdd(out + 0, value.x);
  atomicAdd(out + 1, value.y);
  atomicAdd(out + 2, value.z);
  atomicAdd(out + 3, value.w);
}

struct dirac_diagonal
  : public thrust::unary_function<thrust::tuple<int, const int>, flo::real4>
{
  dirac_diagonal(thrust::device_ptr<const real3> di_vertices,
                 thrust::device_ptr<const int3> di_faces,
                 thrust::device_ptr<const real> di_face_area,
                 thrust::device_ptr<const real> di_rho)
    : di_vertices(std::move(di_vertices.get()))
    , di_faces(std::move(di_faces.get()))
    , di_face_area(std::move(di_face_area.get()))
    , di_rho(std::move(di_rho.get()))
  {
  }

  const real3* di_vertices;
  const int3* di_faces;
  const real* di_face_area;
  const real* di_rho;

  __host__ __device__ flo::real4
  operator()(thrust::tuple<int, const int> id) const
  {
    const int vid = id.get<0>();
    const int fid = id.get<1>();
    // Remove vid from faces[fid]
    int2 edge_idxs;
    {
      const int3 f = di_faces[fid];
      if (f.x == vid)
        edge_idxs = make_int2(f.y, f.z);
      if (f.y == vid)
        edge_idxs = make_int2(f.x, f.z);
      if (f.z == vid)
        edge_idxs = make_int2(f.x, f.y);
    }

    const real3 edge = di_vertices[edge_idxs.y] - di_vertices[edge_idxs.x];
    const real rho = di_rho[vid];
    const real area = di_face_area[fid];

    flo::real4 o_val;
    o_val.x = o_val.y = o_val.z = 0.f;
    o_val.w =
      dot(edge, edge) / (4.f * area) + (rho * rho * area) * reciprocal(9.f);
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
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size =
    (sizeof(flo::real4) + sizeof(uint32_t) * 2) * nthreads_per_block;

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

FLO_API void to_real_quaternion_matrix(
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
  cusp::array1d<real3, cusp::device_memory>::const_view di_vertices,
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<real, cusp::device_memory>::const_view di_rho,
  cusp::array1d<int2, cusp::device_memory>::const_view di_entry_offset,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_cumulative_triangle_valence,
  cusp::array1d<int, cusp::device_memory>::const_view
    di_vertex_triangle_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real4, cusp::device_memory>::view do_dirac_matrix)
{
  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = di_faces.size() * 6 / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size =
    sizeof(flo::real) * block_dim.x * 27 + sizeof(uint32_t) * 6 * block_dim.x;

  // When passing the face and offset data to cuda, we reinterpret them as int
  // arrays. The advantage of this is coalesced memory reads by neighboring
  // threads, and access at a more granular level.
  // The cast is inherently safe due to the alignment of cuda vector types,
  // and reinterpret casting guarantees no changes to the underlying values
  d_intrinsic_dirac_atomic<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices.begin().base().get(),
    reinterpret_cast<const int*>(di_faces.begin().base().get()),
    di_face_area.begin().base().get(),
    di_rho.begin().base().get(),
    reinterpret_cast<const int*>(di_entry_offset.begin().base().get()),
    di_faces.size(),
    do_dirac_matrix.row_indices.begin().base().get(),
    do_dirac_matrix.column_indices.begin().base().get(),
    do_dirac_matrix.values.begin().base().get());
  cudaDeviceSynchronize();

  thrust::counting_iterator<int> counter(0);
  thrust::copy_if(
    counter,
    counter + do_dirac_matrix.values.size(),
    do_diagonals.begin(),
    [d_rows = do_dirac_matrix.row_indices.begin().base().get(),
     d_cols =
       do_dirac_matrix.column_indices.begin().base().get()] __device__(int x) {
      return d_cols[x] + d_rows[x] == 0;
    });

  // Iterator for diagonal matrix entries
  auto diag_begin = thrust::make_permutation_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(do_dirac_matrix.row_indices.begin(),
                         do_dirac_matrix.column_indices.begin())),
    do_diagonals.begin());

  // Generate the diagonal entry, row and column indices
  thrust::tabulate(
    diag_begin, diag_begin + di_vertices.size(), [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });

  // Generate the inverse mapping of vertex triangle adjacency
  // [3, 2, 4] will yield [0,0,0, 1,1, 2,2,2,2]
  thrust::device_vector<int> vert_id(di_cumulative_triangle_valence.back());
  thrust::copy_n(
    thrust::constant_iterator<int>(1),
    di_vertices.size() - 1,
    thrust::make_permutation_iterator(
      vert_id.begin(), di_cumulative_triangle_valence.begin() + 1));
  thrust::inclusive_scan(vert_id.begin(), vert_id.end(), vert_id.begin());

  // Iterate over adjacent faces and the corresponding vertex id
  auto face_vertex_iter = thrust::make_zip_iterator(
    thrust::make_tuple(vert_id.begin(), di_vertex_triangle_adjacency.begin()));

  // Transform opposing edge's, found through the vertex triangle adjacency
  // information, into diagonal dirac contributions
  // Doing this through the iterator saves a memory allocation
  auto dirac_iter =
    thrust::make_transform_iterator(face_vertex_iter,
                                    dirac_diagonal(di_vertices.begin().base(),
                                                   di_faces.begin().base(),
                                                   di_face_area.begin().base(),
                                                   di_rho.begin().base()));

  thrust::reduce_by_key(
    vert_id.begin(),
    vert_id.end(),
    dirac_iter,
    thrust::make_discard_iterator(),
    thrust::make_permutation_iterator(do_dirac_matrix.values.begin(),
                                      do_diagonals.begin()));
}

FLO_DEVICE_NAMESPACE_END

