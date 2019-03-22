#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/histogram.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>
#include <cusp/iterator/strided_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void d_adjacency_matrix_offset(
  const int* __restrict__ di_faces,
  const int* __restrict__ di_vertex_adjacency,
  const int* __restrict__ di_cumulative_valence,
  const int i_nfaces,
  int* __restrict__ do_offset)
{
  const uint fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Determine whether we are calculating a column or row major offset
  // even threads are col major while odd ones are row major
  uint8_t major = threadIdx.y & 1;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uint32_t edge_idx = threadIdx.y >> 1;
  const uchar3 loop = tri_edge_loop(edge_idx);
  // Compute local edge indices rotated by the offset major
  const int2 ep = make_int2(fid * 3 + nth_element(loop, 1 + major),
                            fid * 3 + nth_element(loop, 1 + !major));

  int2 edge = make_int2(di_faces[ep.x], di_faces[ep.y]);

  int begin = di_cumulative_valence[edge.x];
  int end = di_cumulative_valence[edge.x + 1] - 1;
  auto iter = thrust::lower_bound(thrust::seq,
                                  di_vertex_adjacency + begin,
                                  di_vertex_adjacency + end,
                                  edge.y);
  do_offset[fid * 6 + loop.x * 2 + major] = iter - di_vertex_adjacency;
}

}  // namespace

FLO_API int vertex_vertex_adjacency(
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence)
{
  const int max_edges = di_faces.size() * 6;
  auto J = do_adjacency.subarray(0, max_edges);
  auto I = do_adjacency.subarray(max_edges, max_edges);

  // Create vector access to the edge indices
  thrust::device_ptr<int3> I3{reinterpret_cast<int3*>(I.begin().base().get())};
  thrust::device_ptr<int3> J3{reinterpret_cast<int3*>(J.begin().base().get())};

  // Create an iterator that pairs diagonal triplets as they will share the
  // same values
  auto face_edge_iter = thrust::make_zip_iterator(thrust::make_tuple(
    thrust::make_zip_iterator(thrust::make_tuple(I3, J3 + 1)),
    thrust::make_zip_iterator(thrust::make_tuple(I3 + 1, J3))));

  // Create a stride iterator, we will generate 6 edges per face, so need to
  // move 2 triplets at a time
  cusp::strided_iterator<decltype(face_edge_iter)> paired_face_edge_iter(
    face_edge_iter, face_edge_iter + di_faces.size() * 2, 2);

  // For each face output all half edges, e.g. the first pairs 6 should be:
  // I:  0 1 2 1 2 0
  // J:  1 2 0 0 1 2
  // As we can see, the first three of I match the last three of J, and the
  // first three of J match the last three of I, hence our iteration strategy
  thrust::transform(di_faces.begin(),
                    di_faces.end(),
                    paired_face_edge_iter.begin(),
                    [] __device__(int3 f) {
                      auto original = thrust::make_tuple(f, f);
                      auto shuffled = thrust::make_tuple(
                        make_int3(f.y, f.z, f.x), make_int3(f.y, f.z, f.x));
                      return thrust::make_tuple(original, shuffled);
                    });

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
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<int2, cusp::device_memory>::view do_offsets)
{
  //  thrust::device_vector<int2> d_offsets(i_nfaces * 3);
  // Create a view over the faces with more granular access
  cusp::array1d<int, cusp::device_memory>::const_view di_face_vertices{
    thrust::device_ptr<const int>{
      reinterpret_cast<const int*>(di_faces.begin().base().get())},
    thrust::device_ptr<const int>{
      reinterpret_cast<const int*>(di_faces.end().base().get())}};

  // Create a view over the offsets with more granular access
  cusp::array1d<int, cusp::device_memory>::view d_offsets{
    thrust::device_ptr<int>{
      reinterpret_cast<int*>(do_offsets.begin().base().get())},
    thrust::device_ptr<int>{
      reinterpret_cast<int*>(do_offsets.end().base().get())}};

  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  const int nblocks =
    di_faces.size() * 6 / (block_dim.x * block_dim.y * block_dim.z) + 1;

  d_adjacency_matrix_offset<<<nblocks, block_dim>>>(
    di_face_vertices.begin().base().get(),
    di_adjacency.begin().base().get(),
    di_cumulative_valence.begin().base().get(),
    di_faces.size(),
    d_offsets.begin().base().get());
}

FLO_DEVICE_NAMESPACE_END

