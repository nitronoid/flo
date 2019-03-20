#ifndef FLO_DEVICE_INCLUDED_THREAD_UTIL
#define FLO_DEVICE_INCLUDED_THREAD_UTIL

#include "flo/flo_internal.hpp"

FLO_DEVICE_NAMESPACE_BEGIN

__device__ __forceinline__ uint unique_thread_idx1();

__device__ __forceinline__ uint unique_thread_idx2();

__device__ __forceinline__ uint unique_thread_idx3();

__device__ __forceinline__ uint block_index();

__device__ __forceinline__ uint block_volume();

__device__ __forceinline__ uint8_t cycle(uint8_t i_x);

__device__ __forceinline__ uchar3 tri_edge_loop(uint8_t i_e);

__device__ __forceinline__ uchar4 quat_loop(uint8_t i_e);

__device__ __forceinline__ int sign_from_bit(uint8_t i_byte, uint8_t i_bit);

#include "flo/device/thread_util.inl"

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_THREAD_UTIL


