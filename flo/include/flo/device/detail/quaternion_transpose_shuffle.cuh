#ifndef FLO_DEVICE_INCLUDED_DETAIL_QUATERNION_TRANSPOSE_SHUFFLE
#define FLO_DEVICE_INCLUDED_DETAIL_QUATERNION_TRANSPOSE_SHUFFLE

#include "flo/flo_internal.hpp"

FLO_DEVICE_NAMESPACE_BEGIN

namespace detail
{
struct quaternion_transpose_shuffle
{
  quaternion_transpose_shuffle(int x) : w(x)
  {
  }
  int w;

  __host__ __device__ int operator()(int i) const
  {
    // Shuffle in the order:
    // x -> w
    // y -> x
    // z -> y
    // w -> z
    const int32_t x = ((i / w) + 1) & 3;
    const int32_t y = i % w;
    return y * 4 + x;
  }
};

}

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_DETAIL_QUATERNION_TRANSPOSE_SHUFFLE

