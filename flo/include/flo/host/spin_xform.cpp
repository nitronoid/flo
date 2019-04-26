template <typename DerivedV, typename DerivedF, typename DerivedP>
FLO_API void spin_xform(Eigen::MatrixBase<DerivedV>& V,
                        const Eigen::MatrixBase<DerivedF>& F,
                        const Eigen::MatrixBase<DerivedP>& P,
                        const Eigen::SparseMatrix<real>& L)
{
  using namespace Eigen;
  // Calculate the real matrix from our quaternion edges
  auto QL = to_real_quaternion_matrix(L);
  // Remove the final quaternion so we have a positive semi-definite matrix
  QL.conservativeResize(QL.rows() - 4, QL.cols() - 4);

  // Calculate all face areas
  Matrix<real, Dynamic, 1> A;
  igl::doublearea(V, F, A);
  A *= 0.5f;

  // Calculate the valence of every vertex to allocate sparse matrices
  Matrix<int, Dynamic, 1> VV, VA, VAK, VCV;
  vertex_vertex_adjacency(F, VAK, VA, VV, VCV);

  // Calculate the intrinsic dirac operator matrix
  SparseMatrix<real> D;
  intrinsic_dirac(V, F, VV, A, P, D);

  // Calculate the scaling and rotation for our spin transformation
  Matrix<real, Dynamic, 4> X;
  similarity_xform(D, X);

  // Calculate our transformed edges
  Matrix<real, Dynamic, 4> E;
  divergent_edges(V, F, X, L, E);
  // Remove the final edge to ensure we are compatible with the sliced laplacian
  E.conservativeResize(E.rows() - 1, Eigen::NoChange);

  // Solve the final vertex positions
  Matrix<real, Dynamic, Dynamic> NV;
  spin_positions(QL, E, NV);
  NV.conservativeResize(NoChange, 3);
  V = NV;
}

