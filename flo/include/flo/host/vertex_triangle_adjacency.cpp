template <typename DerivedF,
          typename DerivedVTAK,
          typename DerivedVTA,
          typename DerivedVTV,
          typename DerivedVTCV>
FLO_API void
vertex_triangle_adjacency(const Eigen::MatrixBase<DerivedF>& F,
                          Eigen::PlainObjectBase<DerivedVTAK>& VTAK,
                          Eigen::PlainObjectBase<DerivedVTA>& VTA,
                          Eigen::PlainObjectBase<DerivedVTV>& VTV,
                          Eigen::PlainObjectBase<DerivedVTCV>& VTCV)
{
  using namespace Eigen;
  igl::vertex_triangle_adjacency(F, F.maxCoeff() + 1, VTA, VTCV);

  // Get a handle the raw cumulative valence
  auto cv_begin = &VTCV(0);
  // One less valence value than cumulative valence value
  VTV.resize(VTCV.size() - 1);
  // Get a handle the raw valence
  auto v_begin = &VTV(0);
  // Calculate the valence from the cumulative list
  std::adjacent_difference(cv_begin + 1, cv_begin + VTCV.size(), v_begin);

  VTAK.resize(VTA.size());
  for (int i = 0; i < VTV.size(); ++i)
    for (int j = 0; j < VTV(i); ++j)
    {
      VTAK(VTCV(i) + j) = i;
    }
}
