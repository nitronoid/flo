
template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_sort_by_key(RandomAccessIterator1&& i_key_begin,
                       RandomAccessIterator1&& i_key_end,
                       RandomAccessIterator2&& i_new_key_begin,
                       RandomAccessIterator3&&... i_data_begin)
{
  using expand = int[];
  auto new_key_end = i_new_key_begin + (i_key_end - i_key_begin);
  thrust::sequence(i_new_key_begin, new_key_end);
  thrust::sort_by_key(i_key_begin, i_key_end, i_new_key_begin);
  expand{((void)thrust::gather(
            i_new_key_begin, new_key_end, i_data_begin, i_data_begin),
          0)...};
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename... RandomAccessIterator3>
void multi_stable_sort_by_key(RandomAccessIterator1&& i_key_begin,
                              RandomAccessIterator1&& i_key_end,
                              RandomAccessIterator2&& i_new_key_begin,
                              RandomAccessIterator3&&... i_data_begin)
{
  using expand = int[];
  auto new_key_end = i_new_key_begin + (i_key_end - i_key_begin);
  thrust::sequence(i_new_key_begin, new_key_end);
  thrust::stable_sort_by_key(i_key_begin, i_key_end, i_new_key_begin);
  expand{((void)thrust::gather(
            i_new_key_begin, new_key_end, i_data_begin, i_data_begin),
          0)...};
}
