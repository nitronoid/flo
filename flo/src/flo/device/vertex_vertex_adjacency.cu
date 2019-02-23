#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/histogram.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void
d_exhaustive_face_edges(const thrust::device_ptr<const int3> di_faces,
                        const uint i_nfaces,
                        thrust::device_ptr<int> do_I,
                        thrust::device_ptr<int> do_J)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Check we're not out of range
  if (tid >= i_nfaces)
    return;

  const int3 face = di_faces[tid];
  do_I[tid * 6 + 0] = face.z;
  do_J[tid * 6 + 0] = face.x;
  do_I[tid * 6 + 1] = face.x;
  do_J[tid * 6 + 1] = face.z;

  do_I[tid * 6 + 2] = face.x;
  do_J[tid * 6 + 2] = face.y;
  do_I[tid * 6 + 3] = face.y;
  do_J[tid * 6 + 3] = face.x;

  do_I[tid * 6 + 4] = face.y;
  do_J[tid * 6 + 4] = face.z;
  do_I[tid * 6 + 5] = face.z;
  do_J[tid * 6 + 5] = face.y;
}

__global__ void d_adjacency_matrix_offset(
  const thrust::device_ptr<const int> di_faces,
  const thrust::device_ptr<const int> di_vertex_adjacency,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const uint i_nfaces,
  thrust::device_ptr<int> do_offset)
{
  const uint fid = blockIdx.y * blockDim.y + threadIdx.y;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Determine whether we are calculating a column or row major offset
  // even threads are col major while odd ones are row major
  uint8_t major = threadIdx.x & 1;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uint32_t edge_idx = threadIdx.x >> 1;
  const uchar3 loop = edge_loop(edge_idx);
  // Compute local edge indices rotated by the offset major
  const int2 ep = make_int2(fid * 3 + nth_element(loop, major),
                            fid * 3 + nth_element(loop, !major));

  int2 edge = make_int2(di_faces[ep.x], di_faces[ep.y]);

  int begin = di_cumulative_valence[edge.x];
  int end = di_cumulative_valence[edge.x + 1] - 1;
  auto iter = thrust::lower_bound(thrust::seq,
                                  di_vertex_adjacency + begin,
                                  di_vertex_adjacency + end,
                                  edge.y);
  do_offset[fid * 6 + loop.z * 2 + major] = iter - di_vertex_adjacency;
}

}  // namespace

thrust::device_vector<int>
vertex_vertex_adjacency(const thrust::device_ptr<const int3> di_faces,
                        const int i_nfaces,
                        const int i_nvertices,
                        thrust::device_ptr<int> do_valence,
                        thrust::device_ptr<int> do_cumulative_valence)
{
  thrust::device_vector<int> I(i_nfaces * 6);
  thrust::device_vector<int> J(i_nfaces * 6);

  int nthreads = 1024;
  int nblocks = i_nfaces / nthreads + 1;
  d_exhaustive_face_edges<<<nblocks, nthreads>>>(
    di_faces, i_nfaces, I.data(), J.data());

  thrust::sort_by_key(J.begin(), J.end(), I.begin());
  thrust::stable_sort_by_key(I.begin(), I.end(), J.begin());

  auto coord_begin =
    thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin()));

  auto coord_end = thrust::unique_by_key(J.begin(), J.end(), coord_begin);
  int total_valence = coord_end.first - J.begin();
  J.resize(total_valence);

  do_cumulative_valence[0] = 0;
  flo::device::cumulative_dense_histogram_sorted(
    I.data(), do_cumulative_valence + 1, total_valence, i_nvertices);
  flo::device::dense_histogram_from_cumulative(
    do_cumulative_valence + 1, do_valence, i_nvertices);

  return J;
}

thrust::device_vector<int2> adjacency_matrix_offset(
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const int> di_vertex_adjacency,
  const thrust::device_ptr<const int> di_cumulative_valence,
  const int i_nfaces)
{
  thrust::device_vector<int2> d_offsets(i_nfaces * 3);

  dim3 block_dim;
  block_dim.x = 6;
  block_dim.y = 170;
  int nblocks = i_nfaces * 6 / (block_dim.x * block_dim.y * block_dim.z) + 1;
  d_adjacency_matrix_offset<<<nblocks, block_dim>>>(
    thrust::device_ptr<const int>{(const int*)di_faces.get()},
    di_vertex_adjacency,
    di_cumulative_valence,
    i_nfaces,
    thrust::device_ptr<int>{(int*)d_offsets.data().get()});

  return d_offsets;
}

FLO_DEVICE_NAMESPACE_END

