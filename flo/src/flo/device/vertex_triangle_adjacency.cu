#include "flo/device/vertex_triangle_adjacency.cuh"
#include <type_traits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
struct transpose_index : public thrust::unary_function<int, int>
{
  const int width;
  const int height;

  transpose_index(int width, int height) : width(width), height(height)
  {
  }

  __host__ __device__ int operator()(int i) const
  {
    const int x = i % height;
    const int y = i / height;
    return x * width + y;
  }
};
}  // namespace

FLO_API void vertex_triangle_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence)
{
  // Store the number of vertices
  const int nvertices = do_valence.size();
  // Store the number of faces
  const int nfaces = di_faces.num_cols;

  // Map is a transposed view of the adjacency keys
  auto map_it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), transpose_index(3, nfaces));
  auto transposed_it =
    thrust::make_permutation_iterator(do_adjacency_keys.begin(), map_it);
  // Copy the face vertices as the adjacency keys, using transposed layout
  thrust::copy_n(di_faces.values.begin(), nfaces * 3, transposed_it);

  // Create a sequential list of indices mapping adjacency to face vertices
  thrust::tabulate(do_adjacency.begin(),
                   do_adjacency.end(),
                   [] __device__(int i) { return i / 3; });

  // We sort the adjacency using our adjacency keys.
  thrust::stable_sort_by_key(
    do_adjacency_keys.begin(), do_adjacency_keys.end(), do_adjacency.begin());

  // Calculate a dense histogram to find the cumulative valence
  // Create a counting iter to output the index values from the upper_bound
  auto search_begin = thrust::make_counting_iterator(0);
  thrust::upper_bound(do_adjacency_keys.begin(),
                      do_adjacency_keys.end(),
                      search_begin,
                      search_begin + nvertices + 1,
                      do_cumulative_valence.begin());

  // Calculate the non-cumulative valence by subtracting neighboring elements
  thrust::adjacent_difference(do_cumulative_valence.begin(),
                              do_cumulative_valence.end(),
                              do_valence.begin());
}

FLO_DEVICE_NAMESPACE_END
