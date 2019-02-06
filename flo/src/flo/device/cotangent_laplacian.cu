#include "flo/device/cotangent_laplacian.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__device__ uint unique_thread_idx1()
{
  return
    // Global block index
    (blockIdx.x) *
    // Number of threads in a block
    (blockDim.x) + 
    // thread index in block
    (threadIdx.x);
}

__device__ uint unique_thread_idx2()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x) *
    // Number of threads in a block
    (blockDim.x * blockDim.y) + 
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x);
}

__device__ uint unique_thread_idx3()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
    // Number of threads in a block
    (blockDim.x * blockDim.y * blockDim.z) + 
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ uint8_t cycle(uint8_t i_x)
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

__device__ uchar3 edge_loop(uint8_t i_e)
{
  uchar3 loop;
  loop.x = i_e;
  loop.z = cycle(loop.x);
  loop.y = cycle(loop.z);
  return loop;
}
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

  // IMPORTANT, we rotate the thread id's
  const uint tid =
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
    // Number of threads in a block
    (blockDim.x * blockDim.y * blockDim.z) + 
    // thread index in block
    (threadIdx.y + threadIdx.z * blockDim.y + threadIdx.x * blockDim.y * blockDim.z);

  // Check we're not out of range
  if (tid >= i_nfaces * blockDim.y * blockDim.z) return;

  // calculated from the x index only
  const uint f_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Write the face area to shared memory to reduce global memory reads
  if (f_idx < i_nfaces && !threadIdx.y && !threadIdx.z)
  {
    area[threadIdx.x] = di_face_area[f_idx];
  }
  __syncthreads();

  // Get the relevant face
  const int3 f = di_faces[f_idx];

  const uint e_idx = 
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
    // Number of edges in a block
    (blockDim.x * blockDim.z) + 
    // Edge id in this block
    (threadIdx.z + threadIdx.x * blockDim.z);

  // Save the cotangent value into shared memory as multiple threads will,
  // write it into the final matrix
  if (e_idx < i_nfaces * blockDim.z && !threadIdx.y)
  {
    // Get the ordered other vertices
    const uchar3 loop = edge_loop(threadIdx.z);

    // Get positions of vertices
    const real3 v0 = di_vertices[(&f.x)[loop.x]];
    const real3 v1 = di_vertices[(&f.x)[loop.y]];
    const real3 v2 = di_vertices[(&f.x)[loop.z]];

    const real3 e0 = v2 - v1;
    const real3 e1 = v0 - v2;
    const real3 e2 = v1 - v0;
    
    // Value is sum of opposing squared lengths minus this squared length,
    // divided by 4 times the area
    cot_alpha[e_idx] =
      (dot(e1,e1) + dot(e2,e2) - dot(e0,e0)) / (area[threadIdx.x] * 8.f);
  }
  __syncthreads();

  // Will replace this with prettier functions later
  do_I[tid] = (&f.x)[(threadIdx.y & 1u) + 1u];
  do_J[tid] = (&f.x)[((threadIdx.y >> 1u) & 1u) + 1u];
  do_V[tid] = cot_alpha[e_idx];
}
//}


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
  block_dim.y = 3;
  block_dim.z = 4;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = ntriplets / nthreads_per_block + 1;

  size_t shared_memory_size =
    sizeof(real)*block_dim.x*4 + sizeof(uchar2)*block_dim.x*12;

  d_build_triplets<<<nblocks, block_dim, shared_memory_size>>>(
        di_vertices, 
        di_faces, 
        di_face_area, 
        i_nfaces, 
        I.data(), 
        J.data(), 
        V.data());

  return d_L;
}

FLO_DEVICE_NAMESPACE_END

