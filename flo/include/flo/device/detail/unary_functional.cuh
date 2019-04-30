#ifndef FLO_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL
#define FLO_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL

#include "flo/flo_internal.hpp"

FLO_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
struct unary_divides
{
  unary_divides(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs / rhs;
  }
};

template <typename T>
struct unary_multiplies
{
  unary_multiplies(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs * rhs;
  }
};

template <typename T>
struct unary_minus
{
  unary_minus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs - rhs;
  }
};

template <typename T>
struct unary_plus
{
  unary_plus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs + rhs;
  }
};
}

FLO_DEVICE_NAMESPACE_END

#endif // FLO_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL
