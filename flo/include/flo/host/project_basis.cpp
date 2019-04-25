template <typename DerivedV, typename DerivedU, typename BinaryOp>
FLO_API void project_basis(Eigen::MatrixBase<DerivedV>& V,
                           const Eigen::MatrixBase<DerivedU>& U,
                           BinaryOp inner_product)
{
  // Subtract the projected vector from the un-projected
  for (int i = 0; i < U.cols(); ++i)
  {
    V -= inner_product(V, U.col(i)) * U.col(i);
  }
}
