
template <typename T>
void cumulative_dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins)
{
  // Number of bins is the max index value + 1 (0 based indices)
  //const uint nbins = *(di_data + i_ndata-1) + 1;

  // Create a couting iter to output the index values from the upper_bound
  thrust::counting_iterator<T> search_begin(0);
  // Upper bound to find and write the final index for each unique data value,
  // i.e. the end of each bin
  thrust::upper_bound(
      thrust::device, 
      di_data, 
      di_data + i_ndata, 
      search_begin, 
      search_begin + i_nbins, 
      do_histogram);
}

template <typename T>
void cumulative_dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_nbins,
    const uint i_ndata)
{
  // Copy our data
  thrust::device_vector<T> d_data_copy(di_data.size());
  thrust::copy(thrust::device, di_data, di_data + i_ndata, d_data_copy.begin());
  // Sort the copy
  thrust::sort(thrust::device, d_data_copy.begin(), d_data_copy.end());
  // Call our sorted histo function
  cumulative_dense_histogram_sorted(d_data_copy.data(), do_histogram, i_ndata, i_nbins);
}

template <typename T>
void dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins)
{
  cumulative_dense_histogram_sorted(di_data, do_histogram, i_ndata, i_nbins);
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      thrust::device, do_histogram, do_histogram + i_nbins, do_histogram);
}

template <typename T>
void dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata,
    const uint i_nbins)
{
  cumulative_dense_histogram_unsorted(di_data, do_histogram, i_ndata, i_nbins);
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      thrust::device, do_histogram, do_histogram + i_nbins, do_histogram);
}

template <typename T>
void dense_histogram_from_cumulative(
    const thrust::device_ptr<T> di_cumulative,
    thrust::device_ptr<T> do_histogram,
    const uint i_ncumulative)
{
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      thrust::device, di_cumulative, di_cumulative + i_ncumulative, do_histogram);
}

template <typename T>
void cumulative_histogram_from_dense(
    const thrust::device_ptr<T> di_dense,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndense)
{
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::inclusive_scan(
      thrust::device, di_dense, di_dense + i_ndense, do_histogram);
}

template <typename T>
void atomic_histogram(
    const thrust::device_ptr<T> di_data,
    thrust::device_ptr<T> do_histogram,
    const uint i_ndata)
{
  thrust::for_each_n(
      thrust::device,
      di_data,
      i_ndata,
      [do_histogram] __device__(auto x)
      {
        atomicAdd((do_histogram + x).get(), 1);
      });
}

