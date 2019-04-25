template <typename DerivedF,
          typename DerivedVVAK,
          typename DerivedVVA,
          typename DerivedVVV,
          typename DerivedVVCV>
FLO_API void vertex_vertex_adjacency(const Eigen::MatrixBase<DerivedF>& F,
                                     Eigen::PlainObjectBase<DerivedVVAK>& VVAK,
                                     Eigen::PlainObjectBase<DerivedVVA>& VVA,
                                     Eigen::PlainObjectBase<DerivedVVV>& VVV,
                                     Eigen::PlainObjectBase<DerivedVVCV>& VVCV)
{
  using namespace Eigen;
  SparseMatrix<int> A;
  igl::adjacency_matrix(F, A);

  // Get the vertex valences
  VVV.resize(A.cols());
  VVCV.resize(A.cols() + 1);
  VVCV(0) = 0;
  for (int i = 0; i < A.cols(); ++i)
  {
    // Get the valence for this vertex
    VVV(i) = A.col(i).nonZeros();
    // Accumulate the valence for this vertex
    VVCV(i + 1) = VVCV(i) + VVV(i);
  }

  // Get the vertex adjacencies
  VVAK.resize(VVCV(VVCV.size() - 1));
  VVA.resize(VVCV(VVCV.size() - 1));
  for (int k = 0; k < A.outerSize(); ++k)
  {
    int j = 0;
    for (SparseMatrix<int>::InnerIterator it(A, k); it; ++it, ++j)
    {
      VVAK(VVCV(k) + j) = it.col();
      VVA(VVCV(k) + j) = it.row();
    }
  }
}
