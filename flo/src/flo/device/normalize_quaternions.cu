#include "flo/device/center_quaternions.cuh"
#include "flo/device/detail/unary_functional.cuh"
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
struct TupleNorm2
{
  using Tup4 = thrust::tuple<real, real, real, real>;

  __host__ __device__ real operator()(const Tup4& vec) const
  {
    return vec.get<0>() * vec.get<0>() + vec.get<1>() * vec.get<1>() +
           vec.get<2>() * vec.get<2>() + vec.get<3>() * vec.get<3>();
  }
};
}

FLO_API void normalize_quaternions(
    cusp::array2d<flo::real, cusp::device_memory>::view dio_quats)
{
  auto norm_begin = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(
        dio_quats.row(0).begin(),
        dio_quats.row(1).begin(),
        dio_quats.row(2).begin(),
        dio_quats.row(3).begin())),
      TupleNorm2{});
  auto norm_end = norm_begin + dio_quats.num_cols;
  // Find the reciprocal of the largest quaternion norm
  const real rmax = 1.f / std::sqrt(*thrust::max_element(norm_begin, norm_end));
  // Scale all quaternions by the reciprocal norm
  thrust::transform(dio_quats.values.begin(),
                    dio_quats.values.end(),
                    dio_quats.values.begin(),
                    detail::unary_multiplies<real>(rmax));
}

FLO_DEVICE_NAMESPACE_END

