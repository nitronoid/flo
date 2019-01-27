#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/histogram.cuh"

#include <type_traits>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


FLO_DEVICE_NAMESPACE_BEGIN

namespace{
template <typename... Args>
auto make_zip_iterator(Args&&... i_args)
{
  using tuple_t = thrust::tuple<std::decay_t<decltype(i_args)>...>;
  using zip_iter_t = thrust::zip_iterator<tuple_t>;
  return zip_iter_t(thrust::make_tuple(std::forward<Args>(i_args)...));
}
}

void vertex_triangle_adjacency(
    const thrust::device_ptr<int> di_faces,
    const uint i_nfaces,
    const uint i_nverts,
    thrust::device_ptr<int> do_adjacency,
    thrust::device_ptr<int> do_valence,
    thrust::device_ptr<int> do_cumulative_valence)
{
  // Get the number of vertex indices, 3 per face
  const auto nvert_idxs = i_nfaces * 3;
  // The corresponding face index will be the same for all vertices in a face,
  // so we store 0,0,0, 1,1,1, ..., n,n,n
  thrust::tabulate(thrust::device, do_adjacency, do_adjacency + nvert_idxs, 
      [] __device__ (auto idx) { return idx / 3; });

  // Allocate a vector for our vertex indices from the face indices
  thrust::device_vector<int> d_vertex_indices(nvert_idxs);
  thrust::copy_n(thrust::device, di_faces, nvert_idxs, d_vertex_indices.data());

  // Simultaneously sort the two arrays using a zip iterator,
  auto zip_begin = make_zip_iterator(
      d_vertex_indices.begin(), do_adjacency);
  // The sort is based on the vertex indices
  thrust::sort_by_key(
      thrust::device, d_vertex_indices.begin(), d_vertex_indices.end(), zip_begin);
  
  do_cumulative_valence[0] = 0;
  cumulative_dense_histogram_sorted(
      d_vertex_indices.data(), do_cumulative_valence+1, d_vertex_indices.size(), i_nverts);
  dense_histogram_from_cumulative(
      do_cumulative_valence+1, do_valence, i_nverts);
}

FLO_DEVICE_NAMESPACE_END
