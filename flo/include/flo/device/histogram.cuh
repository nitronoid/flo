#ifndef FLO_DEVICE_INCLUDED_HISTOGRAM
#define FLO_DEVICE_INCLUDED_HISTOGRAM

#include "flo/flo_internal.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

FLO_DEVICE_NAMESPACE_BEGIN

template <typename T>
thrust::device_vector<T> cumulative_dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data);

template <typename T>
thrust::device_vector<T> cumulative_dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data);

template <typename T>
thrust::device_vector<T> dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data);

template <typename T>
thrust::device_vector<T> dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data);

template <typename T>
thrust::device_vector<T> dense_histogram_from_cumulative(
    const thrust::device_ptr<T> di_cumulative,
    const uint i_n_cumulative,
    const uint i_n_bins);

#include "flo/device/histogram.inl"//template definitions

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_HISTOGRAM

