
template <typename DerivedV, typename DerivedM, typename DerivedHN>
FLO_API void mean_curvature_normal(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   Eigen::PlainObjectBase<DerivedHN>& HN)
{
  using namespace Eigen;
  Matrix<real, Dynamic, 1> Minv = 1.f / (12.f * M.array());
  HN = (-Minv).asDiagonal() * (2.0f * L * V);
}

template <typename DerivedV, typename DerivedM, typename DerivedH>
FLO_API void mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                            const Eigen::SparseMatrix<real>& L,
                            const Eigen::MatrixBase<DerivedM>& M,
                            Eigen::PlainObjectBase<DerivedH>& H)
{
  using namespace Eigen;
  Matrix<real, Dynamic, 3> HN;
  mean_curvature_normal(V, L, M, HN);
  H = HN.rowwise().norm();
}

template <typename DerivedV,
          typename DerivedM,
          typename DerivedN,
          typename DerivedH>
FLO_API void signed_mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   const Eigen::MatrixBase<DerivedN>& N,
                                   Eigen::PlainObjectBase<DerivedH>& H)
{
  using namespace Eigen;
  Matrix<real, Dynamic, 3> HN;
  mean_curvature_normal(V, L, M, HN);
  H.resize(HN.rows());

  for (int i = 0; i < HN.rows(); ++i)
  {
    // if the angle between the unit and curvature normals is obtuse,
    // we need to flow in the opposite direction, and hence invert our sign
    auto NdotH = -N.row(i).dot(HN.row(i));
    H(i) = std::copysign(HN.row(i).norm(), std::move(NdotH));
  }
}

