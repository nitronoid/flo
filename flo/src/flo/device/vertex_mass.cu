#include "flo/device/vertex_mass.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
struct Divide3
{
  static constexpr real third = 1.f / 3.f;

  __host__ __device__ real operator()(real x) const
  {
    return x * third;
  }
};
}  // namespace

FLO_API void vertex_mass(
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<real, cusp::device_memory>::view do_vertex_mass)
{
  // Generate the inverse mapping of vertex triangle adjacency
  // [3, 2, 4] will yield [0,0,0, 1,1, 2,2,2,2]
  thrust::device_vector<int> vert_id(di_cumulative_valence.back());
  thrust::copy_n(thrust::constant_iterator<int>(1),
                 do_vertex_mass.size() - 1,
                 thrust::make_permutation_iterator(
                   vert_id.begin(), di_cumulative_valence.begin()));
  thrust::inclusive_scan(vert_id.begin(), vert_id.end(), vert_id.begin());

  // Reduce the face areas using the inverse adjacency mapping to lookup by face
  thrust::reduce_by_key(
    vert_id.begin(),
    vert_id.end(),
    thrust::make_permutation_iterator(di_face_area.begin(),
                                      di_adjacency.begin()),
    vert_id.begin(),
    thrust::make_transform_output_iterator(do_vertex_mass.begin(), Divide3{}));
}

FLO_DEVICE_NAMESPACE_END
