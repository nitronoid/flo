#include "flo/device/cotangent_laplacian.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
template <typename T>
__device__ __forceinline__ auto nth_element(const T& i_tuple, uint8_t i_index)
  -> decltype(i_tuple.x)
{
  return (&i_tuple.x)[i_index];
}

__device__ __forceinline__ uint unique_thread_idx1()
{
  return
    // Global block index
    (blockIdx.x) *
      // Number of threads in a block
      (blockDim.x) +
    // thread index in block
    (threadIdx.x);
}

__device__ __forceinline__ uint unique_thread_idx2()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x) *
      // Number of threads in a block
      (blockDim.x * blockDim.y) +
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x);
}

__device__ __forceinline__ uint unique_thread_idx3()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
      // Number of threads in a block
      (blockDim.x * blockDim.y * blockDim.z) +
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x +
     threadIdx.z * blockDim.x * blockDim.y);
}

__device__ __forceinline__ uint block_index()
{
  return blockIdx.x + blockIdx.y * gridDim.x +
         blockIdx.z * gridDim.x * gridDim.y;
}

__device__ __forceinline__ uint block_volume()
{
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ uint8_t cycle(uint8_t i_x)
{
  /************************
     mapping is as follows
     0 -> 2
     1 -> 0
     2 -> 1
  ************************/
  uint8_t c = i_x + 0xFC;
  return __ffs(c) - 1;
}

__device__ __forceinline__ uchar3 edge_loop(uint8_t i_e)
{
  /************************
    e.g. input == 1
     x -> 1
     y -> 2
     z -> 0
  ************************/
  uchar3 loop;
  loop.x = i_e;
  loop.z = cycle(loop.x);
  loop.y = cycle(loop.z);
  return loop;
}

// block dim should be #F*4*3, where #F is some number of faces,
// we have three edges per triangle face, and write four values per edge
__global__ void
d_build_triplets(const thrust::device_ptr<const real3> di_vertices,
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
  real3* points = (real3*)(cached_value + blockDim.x * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* edge_norm2 = (real*)(points + blockDim.x * 3);

  // Block index
  const uint bid = block_index();

  // thread index in block. IMPORTANT, we rotate the thread id's
  const uint tid =
    bid * block_volume() + (threadIdx.y + threadIdx.z * 4 + threadIdx.x * 12);

  // Check we're not out of range
  if (tid >= i_nfaces * 12)
    return;

  // calculated from the x index only
  const uint f_idx = unique_thread_idx1();
  // Get the relevant face
  const int3 f = di_faces[f_idx];

  // Global block index multiplied by number of edges in a block by edge id
  const uint global_edge_idx =
    bid * (blockDim.x * 3) + (threadIdx.z + threadIdx.x * 3);

  // Get the vertex order
  const uchar3 loop = edge_loop(threadIdx.z);
  // Compute local edge indices rotated by the vertex order
  const int e_idx0 = threadIdx.x * 3 + loop.x;
  const int e_idx1 = threadIdx.x * 3 + loop.y;
  const int e_idx2 = threadIdx.x * 3 + loop.z;

  if (f_idx < i_nfaces && !threadIdx.y && !threadIdx.z)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    cached_value[e_idx0] = cached_value[e_idx1] = cached_value[e_idx2] =
      di_face_area[f_idx] * 8.f;
  }
  __syncthreads();

  // Run a thread for each face-edge
  bool edge_thread = global_edge_idx < i_nfaces * 3 && !threadIdx.y;
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
  int source = nth_element(loop, (threadIdx.y & 1u) + 1u);
  int dest = nth_element(loop, ((threadIdx.y >> 1u) & 1u) + 1u);
  // Write the row and column indices
  do_I[tid] = nth_element(f, source);
  do_J[tid] = nth_element(f, dest);
  // If the vertex pair are non identical, our entry should be negative
  do_V[tid] = cached_value[e_idx0] * ((source == dest) * 2 - 1);
}
}  // namespace

FLO_API cusp::coo_matrix<int, real, cusp::device_memory>
cotangent_laplacian(const thrust::device_ptr<const real3> di_vertices,
                    const thrust::device_ptr<const int3> di_faces,
                    const thrust::device_ptr<const real> di_face_area,
                    const int i_nverts,
                    const int i_nfaces,
                    const int i_total_valence)
{
  using SparseMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
  SparseMatrix d_L(i_nverts, i_nverts, i_total_valence + i_nverts);

  const int ntriplets = i_nfaces * 12;
  thrust::device_vector<int> I(ntriplets);
  thrust::device_vector<int> J(ntriplets);
  thrust::device_vector<real> V(ntriplets);

  dim3 block_dim;
  block_dim.x = 1024 / 12;
  block_dim.y = 4;
  block_dim.z = 3;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = ntriplets / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3
  // vertex positions       =>  sizeof(real3) * 3 ==  sizeof(real) * 9
  // edge squared lengths   =>  sizeof(real) * 3
  // === (3 + 9 + 3) * sizeof(real)
  size_t shared_memory_size = sizeof(flo::real) * block_dim.x * 15;

  d_build_triplets<<<nblocks, block_dim, shared_memory_size>>>(di_vertices,
                                                               di_faces,
                                                               di_face_area,
                                                               i_nfaces,
                                                               I.data(),
                                                               J.data(),
                                                               V.data());
  cudaDeviceSynchronize();

  {
    using namespace thrust;
    sort_by_key(
      J.begin(), J.end(), make_zip_iterator(make_tuple(I.begin(), V.begin())));
    sort_by_key(
      I.begin(), I.end(), make_zip_iterator(make_tuple(J.begin(), V.begin())));

    reduce_by_key(make_zip_iterator(make_tuple(I.begin(), J.begin())),
                  make_zip_iterator(make_tuple(I.end(), J.end())),
                  V.begin(),
                  make_zip_iterator(make_tuple(d_L.row_indices.begin(),
                                               d_L.column_indices.begin())),
                  d_L.values.begin(),
                  equal_to<tuple<int, int>>(),
                  plus<float>());
  }

  return d_L;
}

FLO_DEVICE_NAMESPACE_END

