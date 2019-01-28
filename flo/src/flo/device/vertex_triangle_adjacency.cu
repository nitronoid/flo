#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/histogram.cuh"

#include <type_traits>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


FLO_DEVICE_NAMESPACE_BEGIN

void vertex_triangle_adjacency(
    thrust::device_ptr<int> dio_faces,
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
      [] __device__ (int idx) { return idx / 3; });

  // Simultaneously sort the two arrays using a zip iterator,
  auto ptr_tuple = thrust::make_tuple(dio_faces, do_adjacency);
  auto zip_begin = thrust::make_zip_iterator(ptr_tuple);
  // The sort is based on the vertex indices
  thrust::sort_by_key(
      thrust::device, dio_faces, dio_faces + nvert_idxs, zip_begin);
  
  do_cumulative_valence[0] = 0;
  //atomic_histogram(dio_faces, do_valence, nvert_idxs);
  //cumulative_histogram_from_dense(do_valence, do_cumulative_valence + 1, i_nverts);
  cumulative_dense_histogram_sorted(
      dio_faces, do_cumulative_valence+1, nvert_idxs, i_nverts);
  dense_histogram_from_cumulative(
      do_cumulative_valence+1, do_valence, i_nverts);
}

FLO_DEVICE_NAMESPACE_END
