template <int R, int C>
FLO_API void insert_block_sparse(const Eigen::Matrix<real, R, C>& i_block,
                                 Eigen::SparseMatrix<real>& i_mat,
                                 int i_x,
                                 int i_y)
{
  // might be a dynamic matrix so need to get dim from member funcs
  int n_rows = i_block.cols();
  int n_cols = i_block.rows();
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
    {
      i_mat.coeffRef(i_x * n_rows + r, i_y * n_cols + c) = i_block(r, c);
    }
}
inline FLO_API Eigen::SparseMatrix<real>
to_real_quaternion_matrix(const Eigen::SparseMatrix<real>& i_real_matrix)
{
  Eigen::SparseMatrix<real> quat_matrix(i_real_matrix.rows() * 4,
                                        i_real_matrix.cols() * 4);

  Eigen::Matrix<int, Eigen::Dynamic, 1> dim(i_real_matrix.cols() * 4, 1);
  for (int i = 0; i < i_real_matrix.cols(); ++i)
  {
    auto num = 4 * i_real_matrix.col(i).nonZeros();
    dim(i * 4 + 0) = num;
    dim(i * 4 + 1) = num;
    dim(i * 4 + 2) = num;
    dim(i * 4 + 3) = num;
  }
  quat_matrix.reserve(dim);

  using iter_t = Eigen::SparseMatrix<real>::InnerIterator;
  for (int i = 0; i < i_real_matrix.outerSize(); ++i)
  {
    for (iter_t it(i_real_matrix, i); it; ++it)
    {
      auto r = it.row();
      auto c = it.col();

      Eigen::Matrix<real, 4, 1> real_quat(0.f, 0.f, 0.f, it.value());
      auto block = quat_to_block(real_quat);
      insert_block_sparse(block, quat_matrix, r, c);
    }
  }
  return quat_matrix;
}
