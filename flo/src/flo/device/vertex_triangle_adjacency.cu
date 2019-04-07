#include "flo/device/vertex_triangle_adjacency.cuh"
#include <type_traits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include "flo/device/cu_raii.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void vertex_triangle_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  thrust::device_ptr<int> dio_temporary_storage,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence)
{
  // View our temporary storage as an array of ints
  cusp::array1d<int, cusp::device_memory>::view temp{
    dio_temporary_storage, dio_temporary_storage + di_faces.values.size()};

  // Copy the face data asynchronously while we compute the histogram
  ScopedCuStream mem_cpy_stream;
  cudaMemcpyAsync(dio_temporary_storage.get(),
                  di_faces.values.begin().base().get(),
                  sizeof(int) * di_faces.values.size(),
                  cudaMemcpyHostToDevice,
                  mem_cpy_stream);

  // We use atomics to calculate the histogram as the input need not be sorted
  thrust::for_each(di_faces.values.begin(),
                   di_faces.values.end(),
                   [d_valence = do_valence.begin().base().get()] __device__(
                     int x) { atomicAdd(d_valence + x, 1); });

  // Use a prefix scan to calculate offsets into our adjacency list per vertex
  thrust::inclusive_scan(
    do_valence.begin(), do_valence.end(), do_cumulative_valence.begin());

  // Iterate over 0th, 1st, 2nd face indices at once
  const int nfaces = di_faces.num_cols;
  auto adjit = thrust::make_zip_iterator(
    thrust::make_tuple(do_adjacency.begin() + nfaces * 0,
                       do_adjacency.begin() + nfaces * 1,
                       do_adjacency.begin() + nfaces * 2));

  // Create a sequential list of indices mapping adjacency to face vertices
  thrust::tabulate(adjit, adjit + nfaces, [] __device__(int i) {
    return thrust::make_tuple(i, i, i);
  });

  // Make sure the memcpy has finished
  mem_cpy_stream.join();

  // We sort the indices by looking up their face vertex
  thrust::sort_by_key(do_adjacency.begin(), do_adjacency.end(), temp.begin());
  thrust::stable_sort_by_key(temp.begin(), temp.end(), do_adjacency.begin());
}

FLO_DEVICE_NAMESPACE_END
