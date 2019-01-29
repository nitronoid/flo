#ifndef FLO_DEVICE_INCLUDED_HISTOGRAM
#define FLO_DEVICE_INCLUDED_HISTOGRAM

#include "flo/flo_internal.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>

FLO_DEVICE_NAMESPACE_BEGIN

template <typename T>
FLO_API void cumulative_dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins);

template <typename T>
FLO_API void cumulative_dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins);

template <typename T>
FLO_API void dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins);

template <typename T>
FLO_API void dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins);

template <typename T>
FLO_API void dense_histogram_from_cumulative(
    const thrust::device_ptr<T> di_cumulative,
    thrust::device_ptr<T> do_histogram,
    const uint i_n_cumulative);

template <typename T>
FLO_API void cumulative_histogram_from_dense(
    const thrust::device_ptr<T> di_dense,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndense);

template <typename T>
FLO_API void atomic_histogram(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata);

#include "flo/device/histogram.inl"//template definitions

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_HISTOGRAM

