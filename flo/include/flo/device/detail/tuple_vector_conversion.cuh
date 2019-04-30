#ifndef FLO_DEVICE_INCLUDED_DETAIL_TUPLE_VECTOR_CONVERSION
#define FLO_DEVICE_INCLUDED_DETAIL_TUPLE_VECTOR_CONVERSION

#include "flo/flo_internal.hpp"
#include <thrust/tuple.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
struct vector_to_tuple;

template <>
struct vector_to_tuple<real>
{
  using Ret = thrust::tuple<real>;

  __host__ __device__ Ret operator()(real vec) const
  {
    return thrust::make_tuple(vec);
  }
};

template <>
struct vector_to_tuple<real2>
{
  using Ret = thrust::tuple<real, real>;

  __host__ __device__ Ret operator()(real2 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y);
  }
};

template <>
struct vector_to_tuple<real3>
{
  using Ret = thrust::tuple<real, real, real>;

  __host__ __device__ Ret operator()(real3 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y, vec.z);
  }
};

template <>
struct vector_to_tuple<real4>
{
  using Ret = thrust::tuple<real, real, real, real>;

  __host__ __device__ Ret operator()(real4 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y, vec.z, vec.w);
  }
};

template <>
struct vector_to_tuple<int>
{
  using Ret = thrust::tuple<int>;

  __host__ __device__ Ret operator()(int vec) const
  {
    return thrust::make_tuple(vec);
  }
};

template <>
struct vector_to_tuple<int2>
{
  using Ret = thrust::tuple<int, int>;

  __host__ __device__ Ret operator()(int2 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y);
  }
};

template <>
struct vector_to_tuple<int3>
{
  using Ret = thrust::tuple<int, int, int>;

  __host__ __device__ Ret operator()(int3 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y, vec.z);
  }
};

template <>
struct vector_to_tuple<int4>
{
  using Ret = thrust::tuple<int, int, int, int>;

  __host__ __device__ Ret operator()(int4 vec) const
  {
    return thrust::make_tuple(vec.x, vec.y, vec.z, vec.w);
  }
};

}

FLO_DEVICE_NAMESPACE_END

#endif // FLO_DEVICE_INCLUDED_DETAIL_TUPLE_VECTOR_CONVERSION

