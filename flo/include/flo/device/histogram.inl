
template <typename T>
thrust::device_vector<T> cumulative_dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data)
{
  // Number of bins is the max index value + 1 (0 based indices)
  const uint n_bins = *(di_data + i_n_data-1) + 1;

  // Allocate for the histogram
  thrust::device_vector<T> histogram(n_bins);
  // Create a couting iter to output the index values from the upper_bound
  thrust::counting_iterator<T> search_begin(0);
  // Upper bound to find and write the final index for each unique data value,
  // i.e. the end of each bin
  thrust::upper_bound(
      di_data, 
      di_data + i_n_data, 
      search_begin, 
      search_begin + n_bins, 
      histogram.begin());

  return histogram;
}

template <typename T>
thrust::device_vector<T> cumulative_dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data)
{
  // Copy our data
  thrust::device_vector<T> d_data_copy(di_data.size());
  thrust::copy(di_data, di_data + i_n_data, d_data_copy.begin());
  // Sort the copy
  thrust::sort(d_data_copy.begin(), d_data_copy.end());
  // Call our sorted histo function
  return cumulative_dense_histogram_sorted(d_data_copy.data(), i_n_data);
}

template <typename T>
thrust::device_vector<T> dense_histogram_sorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data)
{
  auto histogram = cumulative_dense_histogram_sorted(di_data, i_n_data);
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      histogram.begin(), histogram.end(), histogram.begin());
  return histogram;
}

template <typename T>
thrust::device_vector<T> dense_histogram_unsorted(
    const thrust::device_ptr<T> di_data,
    const uint i_n_data)
{
  auto histogram = cumulative_dense_histogram_unsorted(di_data, i_n_data);
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      histogram.begin(), histogram.end(), histogram.begin());
  return histogram;
}

template <typename T>
thrust::device_vector<T> dense_histogram_from_cumulative(
    const thrust::device_ptr<T> di_cumulative,
    const uint i_n_cumulative,
    const uint i_n_bins)
{
  thrust::device_vector<T> histogram(i_n_bins);
  // Adjacent difference the upper bound to result in the sizes of each bin
  // i.e. the occupancy and final histogram
  thrust::adjacent_difference(
      di_cumulative, di_cumulative + i_n_cumulative, histogram.begin());
  return histogram;
}
