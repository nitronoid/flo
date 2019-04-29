#include "flo/device/adjacency_matrix_indices.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void
d_adjacency_matrix_indices(const int* __restrict__ di_faces,
                           const int* __restrict__ di_vertex_adjacency,
                           const int* __restrict__ di_cumulative_valence,
                           const int i_nfaces,
                           int* __restrict__ do_indices)
{
  const int fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Determine whether we are calculating a column or row major offset
  // even threads are col major while odd ones are row major
  const uint8_t major = threadIdx.y >= 3;

  const uchar3 loop = tri_edge_loop(threadIdx.y - 3 * major);

  // Global vertex indices that make this edge
  const int2 edge =
    make_int2(di_faces[i_nfaces * nth_element(loop, major) + fid],
              di_faces[i_nfaces * nth_element(loop, !major) + fid]);

  int begin = di_cumulative_valence[edge.x];
  int end = di_cumulative_valence[edge.x + 1] - 1;

  auto iter = thrust::lower_bound(thrust::seq,
                                  di_vertex_adjacency + begin,
                                  di_vertex_adjacency + end,
                                  edge.y);
  const int index = (iter - di_vertex_adjacency) + edge.x + (edge.y > edge.x);
  do_indices[i_nfaces * threadIdx.y + fid] = index;
}

}  // namespace

FLO_API void adjacency_matrix_indices(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_indices)
{
  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  const int nblocks =
    di_faces.num_cols * 6 / (block_dim.x * block_dim.y * block_dim.z) + 1;

  d_adjacency_matrix_indices<<<nblocks, block_dim>>>(
    di_faces.values.begin().base().get(),
    di_adjacency.begin().base().get(),
    di_cumulative_valence.begin().base().get(),
    di_faces.num_cols,
    do_indices.values.begin().base().get());
  cudaDeviceSynchronize();
}

FLO_DEVICE_NAMESPACE_END


