#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
}  // namespace

// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void d_cotangent_laplacian_atomic(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const thrust::device_ptr<const int> di_entry_offset,
  const uint i_nfaces,
  thrust::device_ptr<int> do_I,
  thrust::device_ptr<int> do_J,
  thrust::device_ptr<real> do_V)
{
  // Declare one shared memory block
  extern __shared__ real shared_memory[];
  // Create pointers into the block dividing it for the different uses
  real* cached_value = shared_memory;
  // There are is a cached value for each corner of the face so we offset
  real3* points = (real3*)(cached_value + blockDim.y * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* edge_norm2 = (real*)(points + blockDim.y * 3);

  const uint fid = blockIdx.y * blockDim.y + threadIdx.y;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uchar3 loop = edge_loop(threadIdx.x >> 1);

  // Compute local edge indices rotated by the offset major
  const int local_e0 = threadIdx.y * 3 + loop.x;
  const int local_e1 = threadIdx.y * 3 + loop.y;
  const int local_e2 = threadIdx.y * 3 + loop.z;

  if (!loop.x)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    cached_value[local_e0] = cached_value[local_e1] = cached_value[local_e2] =
      di_face_area[fid] * 8.f;
  }
  // Write the vertex positions into shared memory
  if (threadIdx.x & 1)
  {
    points[local_e0] = di_vertices[di_faces[fid*3+loop.x]];
  }
  __syncthreads();
  // Compute squared length of edges and write to shared memory
  if (threadIdx.x & 1)
  {
    const real3 e = points[local_e2] - points[local_e1];
    edge_norm2[local_e0] = dot(e, e);
  }
  __syncthreads();
  if (threadIdx.x & 1)
  {
    real area = cached_value[local_e0];
    // Save the cotangent value into shared memory as multiple threads will,
    // write it into the final matrix
    cached_value[local_e0] =
      (edge_norm2[local_e1] + edge_norm2[local_e2] - edge_norm2[local_e0]) / area;
  }
  __syncthreads();

  int address = di_entry_offset[fid*6+threadIdx.x];
  // Write the row and column indices
  do_I[address] = di_faces[fid*3 + 1 + (threadIdx.x & 1)];
  do_J[address] = di_faces[fid*3 + 1 + !(threadIdx.x & 1)];
  atomicAdd((do_V + address).get(), -cached_value[local_e0]);
}

// block dim should be #F*4*3, where #F is some number of faces,
// we have three edges per triangle face, and write four values per edge
__global__ void d_cotangent_laplacian_triplets(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const uint i_nfaces,
  thrust::device_ptr<int> do_I,
  thrust::device_ptr<int> do_J,
  thrust::device_ptr<real> do_V)
{
  // Declare one shared memory block
  extern __shared__ real shared_memory[];
  // Create pointers into the block dividing it for the different uses
  real* cached_value = shared_memory;
  // There are is a cached value for each corner of the face so we offset
  real3* points = (real3*)(cached_value + blockDim.z * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* edge_norm2 = (real*)(points + blockDim.z * 3);

  // Block index
  const uint bid = block_index();
  // Unique thread index from x, y, z
  const uint tid = unique_thread_idx3();

  // Check we're not out of range
  if (tid >= i_nfaces * 12)
    return;

  // calculated from the z index only
  const uint f_idx = blockIdx.z * blockDim.z + threadIdx.z;
  // Get the relevant face
  const int3 f = di_faces[f_idx];

  // Global block index multiplied by number of edges in a block by edge id
  const uint global_edge_idx =
    bid * (blockDim.z * 3) + (threadIdx.y + threadIdx.z * 3);

  // Get the vertex order
  const uchar3 loop = edge_loop(threadIdx.y);
  // Compute local edge indices rotated by the vertex order
  const int e_idx0 = threadIdx.z * 3 + loop.x;
  const int e_idx1 = threadIdx.z * 3 + loop.y;
  const int e_idx2 = threadIdx.z * 3 + loop.z;

  if (f_idx < i_nfaces && !threadIdx.x && !threadIdx.y)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    cached_value[e_idx0] = cached_value[e_idx1] = cached_value[e_idx2] =
      di_face_area[f_idx] * 8.f;
  }
  // Run a thread for each face-edge
  bool edge_thread = global_edge_idx < i_nfaces * 3 && !threadIdx.x;
  if (edge_thread)
  {
    // Write the vertex positions into shared memory
    points[e_idx0] = di_vertices[nth_element(f, loop.x)];
  }
  __syncthreads();
  if (edge_thread)
  {
    // Compute squared length of edges and write to shared memory
    const real3 e = points[e_idx2] - points[e_idx1];
    edge_norm2[e_idx0] = dot(e, e);
  }
  __syncthreads();
  if (edge_thread)
  {
    real area = cached_value[e_idx0];
    // Save the cotangent value into shared memory as multiple threads will,
    // write it into the final matrix
    cached_value[e_idx0] =
      (edge_norm2[e_idx1] + edge_norm2[e_idx2] - edge_norm2[e_idx0]) / area;
  }
  __syncthreads();

  // Get the vertex index pair to write our value to
  int source = nth_element(loop, (threadIdx.x & 1u) + 1u);
  int dest = nth_element(loop, ((threadIdx.x >> 1u) & 1u) + 1u);
  // Write the row and column indices
  do_I[tid] = nth_element(f, source);
  do_J[tid] = nth_element(f, dest);
  // If the vertex pair are non identical, our entry should be negative
  do_V[tid] = cached_value[e_idx0] * ((source == dest) * 2 - 1);
}
//}  // namespace

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_sort_by_key(RandomAccessIterator1&& i_key_begin,
                       RandomAccessIterator1&& i_key_end,
                       RandomAccessIterator2&& i_new_key_begin,
                       RandomAccessIterator3&&... i_data_begin)
{
  using expand = int[];
  auto new_key_end = i_new_key_begin + (i_key_end - i_key_begin);
  thrust::sequence(i_new_key_begin, new_key_end);
  thrust::sort_by_key(i_key_begin, i_key_end, i_new_key_begin);
  expand{((void)thrust::gather(
            i_new_key_begin, new_key_end, i_data_begin, i_data_begin),
          0)...};
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_stable_sort_by_key(RandomAccessIterator1&& i_key_begin,
                              RandomAccessIterator1&& i_key_end,
                              RandomAccessIterator2&& i_new_key_begin,
                              RandomAccessIterator3&&... i_data_begin)
{
  using expand = int[];
  auto new_key_end = i_new_key_begin + (i_key_end - i_key_begin);
  thrust::sequence(i_new_key_begin, new_key_end);
  thrust::stable_sort_by_key(i_key_begin, i_key_end, i_new_key_begin);
  expand{((void)thrust::gather(
            i_new_key_begin, new_key_end, i_data_begin, i_data_begin),
          0)...};
}

FLO_API cusp::coo_matrix<int, real, cusp::device_memory>
cotangent_laplacian(const thrust::device_ptr<const real3> di_vertices,
                    const thrust::device_ptr<const int3> di_faces,
                    const thrust::device_ptr<const real> di_face_area,
                    const thrust::device_ptr<const int> di_cumulative_valence,
                    const thrust::device_ptr<const int2> di_entry_offset,
                    const int i_nverts,
                    const int i_nfaces,
                    const int i_total_valence)
{
  using SparseMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
  SparseMatrix d_L(i_nverts, i_nverts, i_total_valence);
  thrust::fill(d_L.values.begin(), d_L.values.end(), 0);

  dim3 block_dim;
  block_dim.x = 6;
  block_dim.y = 170;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = i_nfaces*6 / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size = sizeof(flo::real) * block_dim.y * 15;

  d_cotangent_laplacian_atomic<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices,
    thrust::device_ptr<const int>{(const int*)di_faces.get()},
    di_face_area,
    di_cumulative_valence,
    thrust::device_ptr<const int>{(const int*)di_entry_offset.get()},
    i_nfaces,
    d_L.column_indices.data(),
    d_L.row_indices.data(),
    d_L.values.data());
  cudaDeviceSynchronize();

  return d_L;
}

FLO_API cusp::coo_matrix<int, real, cusp::device_memory>
cotangent_laplacian(const thrust::device_ptr<const real3> di_vertices,
                    const thrust::device_ptr<const int3> di_faces,
                    const thrust::device_ptr<const real> di_face_area,
                    const int i_nverts,
                    const int i_nfaces,
                    const int i_total_valence)
{
  const int ntriplets = i_nfaces * 12;
  thrust::device_vector<int> I(ntriplets);
  thrust::device_vector<int> J(ntriplets);
  thrust::device_vector<real> V(ntriplets);

  dim3 block_dim;
  block_dim.x = 4;
  block_dim.y = 3;
  block_dim.z = 64;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = ntriplets / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size = sizeof(flo::real) * block_dim.z * 15;

  d_cotangent_laplacian_triplets<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices,
    di_faces,
    di_face_area,
    i_nfaces,
    I.data(),
    J.data(),
    V.data());
  cudaDeviceSynchronize();

  auto coord_begin =
    thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin()));
  auto coord_end = coord_begin + ntriplets;

  thrust::device_vector<int> seq(ntriplets);
  multi_sort_by_key(J.begin(), J.end(), seq.begin(), I.begin(), V.begin());
  multi_stable_sort_by_key(
    I.begin(), I.end(), seq.begin(), J.begin(), V.begin());

  using SparseMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
  SparseMatrix d_L(
    i_nverts, i_nverts, i_total_valence + i_nverts);  // num_entries + 1);

  thrust::reduce_by_key(coord_begin,
                        coord_end,
                        V.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                          d_L.row_indices.begin(), d_L.column_indices.begin())),
                        d_L.values.begin(),
                        thrust::equal_to<thrust::tuple<int, int>>(),
                        thrust::plus<real>());

  return d_L;
}

FLO_DEVICE_NAMESPACE_END

