template <typename DerivedV, typename BinaryOp, typename DerivedU>
FLO_API void orthonormalize(const Eigen::MatrixBase<DerivedV>& V,
                            BinaryOp inner_product,
                            Eigen::PlainObjectBase<DerivedU>& U)
{
  using namespace Eigen;
  // Normalize is defined using a self inner product
  auto normalize = [&](const Matrix<real, Dynamic, 1>& x) {
    return x.array() / std::sqrt(inner_product(x, x));
  };
  // Dimensionality of vectors
  const auto nvectors = V.cols();
  const auto dim = V.rows();

  // Allocate space for our final basis matrix
  U.resize(dim, nvectors);

  // The first u0 is v0 normalized
  U.col(0) = normalize(V.col(0));
  // Gramm Schmit process
  for (int i = 1; i < nvectors; ++i)
  {
    U.col(i) = V.col(i) - inner_product(V.col(i), U.col(0)) * U.col(0);
    for (int k = 1; k < i; ++k)
    {
      U.col(i) -= inner_product(U.col(i), U.col(k)) * U.col(k);
    }
    U.col(i) = normalize(U.col(i).eval());
  }
}

