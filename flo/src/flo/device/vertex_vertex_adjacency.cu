#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/histogram.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>
#include <cusp/iterator/strided_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void
d_adjacency_matrix_offset(const int* __restrict__ di_faces,
                          const int* __restrict__ di_vertex_adjacency,
                          const int* __restrict__ di_cumulative_valence,
                          const int i_nfaces,
                          int* __restrict__ do_offset)
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
  const int offset = (iter - di_vertex_adjacency) + edge.x + (edge.y > edge.x);
  do_offset[i_nfaces * threadIdx.y + fid] = offset;
}

}  // namespace

FLO_API int vertex_vertex_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence)
{
  // Number of faces is equal to length of one column
  const int nfaces = di_faces.num_cols;
  // 3 edges per face
  const int nedges = nfaces * 3;

  // Create views over the face vertex columns
  auto face_vertex_0 = di_faces.row(0);
  auto face_vertex_1 = di_faces.row(1);
  auto face_vertex_2 = di_faces.row(2);

  // Reduce the verbosity
  auto J = do_adjacency;
  auto I = do_adjacency_keys;

  // TODO: Copy asynchronously
  // Copy our columns
  // Copies 0,1,2
  thrust::copy(face_vertex_0.begin(), face_vertex_2.end(), I.begin());
  // Copies 0,1,2,1,2
  thrust::copy(face_vertex_1.begin(), face_vertex_2.end(), I.begin() + nedges);
  // Copies 0,1,2,1,2,0
  thrust::copy(
    face_vertex_0.begin(), face_vertex_0.end(), I.begin() + nfaces * 5);

  // Copies 1,2
  thrust::copy(face_vertex_1.begin(), face_vertex_2.end(), J.begin());
  // Copies 1,2,0
  thrust::copy(
    face_vertex_0.begin(), face_vertex_0.end(), J.begin() + nfaces * 2);
  // Copies 1,2,0,0,1,2
  thrust::copy(face_vertex_0.begin(), face_vertex_2.end(), J.begin() + nedges);

  // We now have:
  // I:  0 1 2 1 2 0
  // J:  1 2 0 0 1 2

  // Sort by column and then row to cluster all adjacency by the key vertex
  thrust::sort_by_key(J.begin(), J.end(), I.begin());
  thrust::stable_sort_by_key(I.begin(), I.end(), J.begin());

  // Remove all duplicate edges
  auto coord_begin =
    thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin()));
  auto coord_end = thrust::unique_by_key(J.begin(), J.end(), coord_begin);
  const int total_valence = coord_end.first - J.begin();

  // Calculate a dense histogram to find the cumulative valence
  // Create a counting iter to output the index values from the upper_bound
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(I.begin(),
                      I.begin() + total_valence,
                      search_begin,
                      search_begin + do_cumulative_valence.size(),
                      do_cumulative_valence.begin());

  // Calculate the non-cumulative valence by subtracting neighbouring elements
  thrust::adjacent_difference(do_cumulative_valence.begin(),
                              do_cumulative_valence.end(),
                              do_valence.begin());

  // Return the final size of the adjacency list
  return total_valence;
}

FLO_API void adjacency_matrix_offset(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_offsets)
{
  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  const int nblocks =
    di_faces.num_cols * 6 / (block_dim.x * block_dim.y * block_dim.z) + 1;

  d_adjacency_matrix_offset<<<nblocks, block_dim>>>(
    di_faces.values.begin().base().get(),
    di_adjacency.begin().base().get(),
    di_cumulative_valence.begin().base().get(),
    di_faces.num_cols,
    do_offsets.values.begin().base().get());
}

FLO_DEVICE_NAMESPACE_END

