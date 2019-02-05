#include "flo/device/cotangent_laplacian.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{

__device__ uint unique_thread_idx(
    const dim3&  k_grid_dim, 
    const uint3& k_block_idx, 
    const dim3&  k_block_dim, 
    const uint3& k_thread_idx)
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
    // Number of threads in a block
    (blockDim.x * blockDim.y * blockDim.z) + 
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

// Pairs of face vert indices for triangle edges,
// corresponding to our matrix entries
__constant__ uchar2 dk_vertex_pairs[12];

// block dim should be M*3*4, where M is some number of faces,
// we have three edges per triangle face, and write four values per edge
__global__ void d_build_triplets(
    const thrust::device_ptr<const int3> di_faces,
    const thrust::device_ptr<const real> di_C,
    const uint i_nentries,
    thrust::device_ptr<int> do_I,
    thrust::device_ptr<int> do_J,
    thrust::device_ptr<real> do_V)
{
  const uint tid = unique_thread_idx(gridDim, blockIdx, blockDim, threadIdx);

  // Check we're not out of range
  if (tid >= i_nentries) return;

  const uint f_idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Get the relavent face, calculated from the x index only
  const int3 f = di_faces[f_idx];

  // Get the source and dest vertex indices for this edge
  const uchar2 vpair = dk_vertex_pairs[threadIdx.y * blockDim.z + threadIdx.z];

  // Will replace this with prettier functions later
  do_I[tid] = *((&f.x) + vpair.x);
  do_J[tid] = *((&f.x) + vpair.y);
  do_V[tid] = di_C[f_idx*3 + threadIdx.y];
}

__global__ void d_calculate_cotangent_values(
    const thrust::device_ptr<const int3> di_faces,
    const thrust::device_ptr<const real3> di_vertices,
    const thrust::device_ptr<const real> di_face_area,
    const uint i_nfaces,
    thrust::device_ptr<real> do_C)
{
  const uint tid = unique_thread_idx(gridDim, blockIdx, blockDim, threadIdx);

  // Check we're not out of range
  if (tid >= i_nfaces * 3) return;

  // reuse const memory
  // Get the source and dest vertex indices for this edge
  const uchar2 vpair = dk_vertex_pairs[threadIdx.y];

  do_C[tid] = /*calc here*/0;
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
  thrust::device_vector<real3> C(ntriplets);


  return d_L;
}

FLO_DEVICE_NAMESPACE_END

