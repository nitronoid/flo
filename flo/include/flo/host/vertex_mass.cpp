template <typename DerivedV, typename DerivedF, typename DerivedM>
FLO_API void vertex_mass(const Eigen::MatrixBase<DerivedV>& V,
                         const Eigen::MatrixBase<DerivedF>& F,
                         Eigen::PlainObjectBase<DerivedM>& M)
{
  using namespace Eigen;
  M.resize(V.rows());
  M.setConstant(0.0f);
  // Calculate all face areas
  Matrix<real, Dynamic, 1> A;
  igl::doublearea(V, F, A);

  // For every face
  for (int i = 0; i < F.rows(); ++i)
  {
    const auto& f = F.row(i);
    constexpr auto sixth = 1.f / 6.f;
    const auto thirdArea = A(i) * sixth;

    M(f(0)) += thirdArea;
    M(f(1)) += thirdArea;
    M(f(2)) += thirdArea;
  }
}
