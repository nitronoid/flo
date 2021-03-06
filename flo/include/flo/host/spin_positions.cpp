template <typename DerivedQE, typename DerivedV>
FLO_API void spin_positions(const Eigen::SparseMatrix<real>& QL,
                            const Eigen::MatrixBase<DerivedQE>& QE,
                            Eigen::PlainObjectBase<DerivedV>& V)
{
  using namespace Eigen;
  // Solve for our new positions
  // If double precision, use Cholmod solver
  // Cholmod not supported for single precision
  SimplicialLLT<SparseMatrix<real>, Lower> cg;
  cg.compute(QL);

  Eigen::Matrix<real, Dynamic, 4, RowMajor> QEr = QE;

  for (int i = 0; i < QEr.size() / 4; ++i)
  {
    const real z = QEr(i * 4 + 3);
    QEr(i * 4 + 3) = QEr(i * 4 + 2);
    QEr(i * 4 + 2) = QEr(i * 4 + 1);
    QEr(i * 4 + 1) = QEr(i * 4 + 0);
    QEr(i * 4 + 0) = z;
  }
  Map<Matrix<real, Dynamic, 1>> b(QEr.data(), QEr.size());
  Matrix<real, Dynamic, 1> flat = cg.solve(b);

  V.resize((flat.size() / 4), 4);
  for (int i = 0; i < flat.size() / 4; ++i)
  {
    const real z = flat(i * 4 + 0);
    V.row(i)(0) = flat(i * 4 + 1);
    V.row(i)(1) = flat(i * 4 + 2);
    V.row(i)(2) = flat(i * 4 + 3);
    V.row(i)(3) = z;
  }

  // Remove the mean to center the positions
  const Eigen::Matrix<flo::real, 1, 4, Eigen::RowMajor> average =
    V.colwise().sum().array() / V.rows();
  V.rowwise() -= average;
  // Normalize positions
  const real max_dist = std::sqrt(V.rowwise().squaredNorm().maxCoeff());
  V *= (1.f / max_dist);
}
