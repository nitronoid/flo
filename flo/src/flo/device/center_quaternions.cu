#include "flo/device/center_quaternions.cuh"
#include "flo/device/detail/unary_functional.cuh"
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void center_quaternions(
    cusp::array2d<flo::real, cusp::device_memory>::view dio_quats)
{
  auto count_it = thrust::make_counting_iterator(0);
  auto discard_it = thrust::make_discard_iterator();
  // Iterator that provides the row of a quaternion element
  auto row_begin = thrust::make_transform_iterator(
    count_it, detail::unary_divides<int>(dio_quats.num_cols));
  auto row_end = row_begin + dio_quats.num_entries;
  // Find the sum of each quaternion element
  thrust::device_vector<real> sum(4);
  thrust::reduce_by_key(
    row_begin, row_end, dio_quats.values.begin(), discard_it, sum.begin());
  // On read divide by the number of quaternions to find an average
  auto avg_it = thrust::make_transform_iterator(
    sum.begin(), detail::unary_multiplies<real>(1.f / dio_quats.num_cols));
  // Subtract the average from each quaternion
  thrust::transform(
      dio_quats.values.end(),
      dio_quats.values.end(),
      thrust::make_permutation_iterator(sum.begin(), row_begin),
      dio_quats.values.begin(),
      thrust::minus<flo::real>());
}

FLO_DEVICE_NAMESPACE_END
