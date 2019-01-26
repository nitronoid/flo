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
    const thrust::device_vector<int>& di_faces,
    const uint i_n_verts,
    thrust::device_vector<int>& dio_adjacency,
    thrust::device_vector<int>& dio_valence,
    thrust::device_vector<int>& dio_cumulative_valence)
{
  // Get the number of vertex indices, 3 per face
  const auto n_vert_idxs = di_faces.size();
  // Allocate a vector of face indices on the gpu
  thrust::device_vector<int> d_face_indices(n_vert_idxs);
  // The corresponding face index will be the same for all vertices in a face,
  // so we store 0,0,0, 1,1,1, ..., n,n,n
  thrust::tabulate(d_face_indices.begin(), d_face_indices.end(), 
      [] __device__ (auto idx) { return idx / 3; });

  // Allocate a vector for our vertex indices from the face indices
  thrust::device_vector<int> d_vertex_indices = di_faces;

  // Simultaneously sort the two arrays using a zip iterator,
  auto zip_begin = make_zip_iterator(
      d_vertex_indices.begin(), d_face_indices.begin());
  // The sort is based on the vertex indices
  thrust::sort_by_key(d_vertex_indices.begin(), d_vertex_indices.end(), zip_begin);
  
  auto offset = cumulative_dense_histogram_sorted(
      d_vertex_indices.data(), d_vertex_indices.size());
  auto vertex_face_valence = dense_histogram_from_cumulative(
      offset.data(), offset.size(), i_n_verts);

  // Copy the vertex face adjacency
  thrust::copy_n(d_face_indices.begin(), n_vert_idxs, dio_adjacency.begin());
  // Copy the cumulative valence, first is zero
  dio_cumulative_valence[0] = 0;
  thrust::copy_n(offset.begin(), i_n_verts, dio_cumulative_valence.begin() + 1);
  // Copy the vertex face valence
  thrust::copy_n(vertex_face_valence.begin(), i_n_verts, dio_valence.begin());
}

FLO_DEVICE_NAMESPACE_END
