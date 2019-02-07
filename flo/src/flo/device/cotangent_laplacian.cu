#include "flo/device/cotangent_laplacian.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>


FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
template <typename T>
__device__ __forceinline__ auto 
nth_element(const T& i_tuple, uint8_t i_index) -> decltype(i_tuple.x)
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
    (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
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
    return  __ffs(c) - 1;
}

__device__ __forceinline__ uchar3 edge_loop(uint8_t i_e)
{
  uchar3 loop;
  loop.x = i_e;
  loop.z = cycle(loop.x);
  loop.y = cycle(loop.z);
  return loop;
}

// block dim should be 4*3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write four values per edge
__global__ void d_build_triplets(
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
  real* area = shared_memory;
  // There are nfaces number of areas so we offset
  real* cot_alpha = area + blockDim.x;
  // There are is a cot value for each corner of the face so we offset
  real3* points = (real3*)(cot_alpha + blockDim.x*3);

  // Block index
  const uint block_index =
    blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  // Number of threads in a block
  const uint n_threads_in_block = blockDim.x * blockDim.y * blockDim.z;

  const uint tid = block_index * n_threads_in_block +
    // thread index in block. IMPORTANT, we rotate the thread id's
    (threadIdx.y + threadIdx.z * blockDim.y + threadIdx.x * blockDim.y * blockDim.z);

  // Check we're not out of range
  if (tid >= i_nfaces * blockDim.y * blockDim.z) return;

  // calculated from the x index only
  const uint f_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Get the relevant face
  const int3 f = di_faces[f_idx];

  // Write the face area, and vertex positions in to shared memory,
  // we do this to reduce global memory reads
  if (f_idx < i_nfaces && !threadIdx.y && !threadIdx.z)
  {
    area[threadIdx.x] = di_face_area[f_idx];
    points[threadIdx.x*3 + 0] = di_vertices[f.x];
    points[threadIdx.x*3 + 1] = di_vertices[f.y];
    points[threadIdx.x*3 + 2] = di_vertices[f.z];
  }
  __syncthreads();


  const uint e_idx = 
    // Global block index multiplied by number of edges in a block
    block_index * (blockDim.x * blockDim.z) + 
    // Edge id in this block
    (threadIdx.z + threadIdx.x * blockDim.z);

  // Save the cotangent value into shared memory as multiple threads will,
  // write it into the final matrix
  // Get the ordered other vertices
  const uchar3 loop = edge_loop(threadIdx.z);
  if (e_idx < i_nfaces * blockDim.z && !threadIdx.y)
  {
    // Get positions of vertices
    const real3 v0 = points[threadIdx.x*3 + loop.x];
    const real3 v1 = points[threadIdx.x*3 + loop.y];
    const real3 v2 = points[threadIdx.x*3 + loop.z];

    // Get the edge vectors for the face
    const real3 e0 = v2 - v1;
    const real3 e1 = v0 - v2;
    const real3 e2 = v1 - v0;
    
    // Value is sum of opposing squared lengths minus this squared length,
    // divided by 4 times the area
    cot_alpha[e_idx] =
      (dot(e1,e1) + dot(e2,e2) - dot(e0,e0)) / (area[threadIdx.x] * 8.f);
  }
  __syncthreads();

  // Get the vertex index pair to write our value to 
  uint8_t source = nth_element(loop,  (threadIdx.y        & 1u) + 1u);
  uint8_t dest   = nth_element(loop, ((threadIdx.y >> 1u) & 1u) + 1u);
  // If the pair are non identical, our entry should be negative
  int8_t sign = (source == dest)*2 - 1;
  // Write the row and column indices
  do_I[tid] = nth_element(f, source);
  do_J[tid] = nth_element(f, dest);
  // Write the signed value
  do_V[tid] = cot_alpha[e_idx] * sign;
}
}


FLO_API cusp::coo_matrix<int, real, cusp::device_memory> cotangent_laplacian(
    const thrust::device_ptr<const real3> di_vertices,
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
  size_t shared_memory_size = sizeof(flo::real)*block_dim.x*7;


  d_build_triplets<<<nblocks, block_dim, shared_memory_size>>>(
        di_vertices, 
        di_faces, 
        di_face_area, 
        i_nfaces, 
        I.data(), 
        J.data(), 
        V.data());
 
  {
    using namespace thrust;
    stable_sort_by_key(J.begin(), J.end(), make_zip_iterator(make_tuple(I.begin(), V.begin())));
    stable_sort_by_key(I.begin(), I.end(), make_zip_iterator(make_tuple(J.begin(), V.begin())));

    reduce_by_key(
        make_zip_iterator(make_tuple(I.begin(), J.begin())),
        make_zip_iterator(make_tuple(I.end(), J.end())),
        V.begin(),
        make_zip_iterator(make_tuple(d_L.row_indices.begin(), d_L.column_indices.begin())),
        d_L.values.begin(),
        equal_to<tuple<int,int>>(),
        plus<float>());
  }

  return d_L;
}

FLO_DEVICE_NAMESPACE_END

