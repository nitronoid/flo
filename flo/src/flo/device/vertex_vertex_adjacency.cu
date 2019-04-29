#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/find.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
struct TupleEqual
{
  __host__ __device__ bool operator()(const thrust::tuple<int, int>& lhs,
                                      const thrust::tuple<int, int>& rhs)
  {
    return (lhs.get<0>() == rhs.get<0>()) && (lhs.get<1>() == rhs.get<1>());
  }
};
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
  auto entry_begin =
    thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin()));
  auto entry_end = thrust::unique_by_key(
    entry_begin, entry_begin + nedges * 2, entry_begin, TupleEqual{});
  const int total_valence = entry_end.first - entry_begin;

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

FLO_DEVICE_NAMESPACE_END

