#ifndef FLO_DEVICE_INCLUDED_MULTI_SORT
#define FLO_DEVICE_INCLUDED_MULTI_SORT

#include "flo/flo_internal.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

FLO_DEVICE_NAMESPACE_BEGIN

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_sort_by_key(RandomAccessIterator1&& i_key_begin,
                       RandomAccessIterator1&& i_key_end,
                       RandomAccessIterator2&& i_new_key_begin,
                       RandomAccessIterator3&&... i_data_begin);

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_stable_sort_by_key(RandomAccessIterator1&& i_key_begin,
                              RandomAccessIterator1&& i_key_end,
                              RandomAccessIterator2&& i_new_key_begin,
                              RandomAccessIterator3&&... i_data_begin);

#include "flo/device/multi_sort.inl"  //template definitions

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_MULTI_SORT
