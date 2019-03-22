#include "flo/device/vertex_triangle_adjacency.cuh"
#include <type_traits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void vertex_triangle_adjacency(
  cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
  thrust::device_ptr<void> dio_temporary_storage,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence)
{
  // Create a view over the faces with more granular access
  cusp::array1d<int, cusp::device_memory>::const_view di_face_vertices{
    thrust::device_ptr<const int>{
      reinterpret_cast<const int*>(di_faces.begin().base().get())},
    thrust::device_ptr<const int>{
      reinterpret_cast<const int*>(di_faces.end().base().get())}};

  // We use atomics to calculate the histogram as the input need not be sorted
  thrust::for_each(di_face_vertices.begin(),
                   di_face_vertices.end(),
                   [d_valence = do_valence.begin().base().get()] __device__(
                     int x) { atomicAdd(d_valence + x, 1); });

  // Use a prefix scan to calculate offsets into our adjacency list per vertex
  thrust::inclusive_scan(
    do_valence.begin(), do_valence.end(), do_cumulative_valence.begin());

  // Create a sequential list of indices mapping adjacency to face vertices
  thrust::sequence(do_adjacency.begin(), do_adjacency.end());

  // View our temporary storage as an array of ints
  cusp::array1d<int, cusp::device_memory>::view temp{
    thrust::device_ptr<int>{static_cast<int*>(dio_temporary_storage.get())},
    thrust::device_ptr<int>{static_cast<int*>(dio_temporary_storage.get()) +
                            di_face_vertices.size()}};

  // Make a copy of the temporary storage
  thrust::copy(di_face_vertices.begin(), di_face_vertices.end(), temp.begin());

  // We sort the indices by looking up their face vertex
  thrust::sort_by_key(temp.begin(), temp.end(), do_adjacency.begin());

  // Finally we simply divide the indices by three to retrieve the face index,
  // as we know there are 3 vertices per face.
  thrust::transform(do_adjacency.begin(),
                    do_adjacency.end(),
                    do_adjacency.begin(),
                    [] __device__(int x) { return x / 3; });
}

FLO_DEVICE_NAMESPACE_END
