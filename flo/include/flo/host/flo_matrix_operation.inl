template <int R, int C>
FLO_API void insert_block_sparse(
    const Eigen::Matrix<double, R, C>& i_block,
    Eigen::SparseMatrix<double>& i_mat,
    uint i_x,
    uint i_y)
{
  // might be a dynamic matrix so need to get dim from member funcs
  uint n_rows = i_block.cols();
  uint n_cols = i_block.rows();
  for (uint r = 0u; r < n_rows; ++r)
  for (uint c = 0u; c < n_cols; ++c)
  {
    i_mat.coeffRef(i_x * n_rows + r, i_y * n_cols + c) = i_block(r, c);
  }
}

template <int R, int C, int IR, int IC>
FLO_API void insert_block_dense(
    const Eigen::Matrix<double, R, C>& i_block,
    Eigen::Matrix<double, IR, IC>& i_mat,
    uint i_x,
    uint i_y)
{
  // might be a dynamic matrix so need to get dim from member funcs
  uint n_rows = i_block.cols();
  uint n_cols = i_block.rows();
  for (uint c = 0u; c < n_cols; ++c)
  for (uint r = 0u; r < n_rows; ++r)
  {
    i_mat(i_x*n_rows + r, i_y*n_cols + c) = i_block(r, c);
  }
}

template <typename T, int R, int C>
FLO_API std::vector<Eigen::Matrix<T, C, 1>> matrix_to_array(
    const Eigen::Matrix<T, R, C>& i_mat)
{
  std::vector<Eigen::Matrix<T, C, 1>> ret;
  ret.reserve(i_mat.rows());

  for (int i = 0; i < i_mat.rows(); ++i)
  {
    ret.emplace_back(i_mat.row(i));
  }

  return ret;
}

template <typename T, int R>
FLO_API std::vector<T> matrix_to_array(
    const Eigen::Matrix<T, R, 1>& i_mat)
{
  std::vector<T> ret;
  ret.reserve(i_mat.rows());

  for (int i = 0; i < i_mat.rows(); ++i)
  {
    ret.push_back(i_mat(i));
  }

  return ret;
}
