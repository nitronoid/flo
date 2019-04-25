template <typename DerivedV,
          typename DerivedF,
          typename DerivedH,
          typename DerivedE>
FLO_API void divergent_edges(const Eigen::MatrixBase<DerivedV>& V,
                             const Eigen::MatrixBase<DerivedF>& F,
                             const Eigen::MatrixBase<DerivedH>& h,
                             const Eigen::SparseMatrix<real>& L,
                             Eigen::PlainObjectBase<DerivedE>& E)
{
  using namespace Eigen;
  E.resize(V.rows(), 4);
  E.setConstant(0.f);
  // For every face
  for (int k = 0; k < F.rows(); ++k)
  {
    const auto& f = F.row(k);
    // For each edge in the face
    for (int i = 0; i < 3; ++i)
    {
      int a = f((i + 1) % 3);
      int b = f((i + 2) % 3);
      if (a > b)
        std::swap(a, b);

      const auto& l1 = h.row(a);
      const auto& l2 = h.row(b);

      Matrix<real, 3, 1> edge = V.row(a) - V.row(b);
      Matrix<real, 4, 1> e(edge[0], edge[1], edge[2], 0.f);

      constexpr auto third = 1.f / 3.f;
      constexpr auto sixth = 1.f / 6.f;

      Matrix<real, 4, 1> et =
        hammilton_product(hammilton_product(third * conjugate(l1), e), l1) +
        hammilton_product(hammilton_product(sixth * conjugate(l1), e), l2) +
        hammilton_product(hammilton_product(sixth * conjugate(l2), e), l1) +
        hammilton_product(hammilton_product(third * conjugate(l2), e), l2);

      auto cot_alpha = L.coeff(a, b) * 0.5f;
      E.row(a) -= et * cot_alpha;
      E.row(b) += et * cot_alpha;
    }
  }
}

