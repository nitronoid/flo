template <typename DerivedX>
FLO_API void similarity_xform(const Eigen::SparseMatrix<real>& D,
                              Eigen::PlainObjectBase<DerivedX>& X,
                              int back_substitutions)
{
  using namespace Eigen;
  // Calculate the length of our matrix,
  // and hence the number of quaternions we should expect
  const int vlen = D.cols();
  const int qlen = vlen / 4;

  SimplicialLLT<SparseMatrix<real>, Lower> cg;
  cg.compute(D);

  // Init every real part to 1, all imaginary parts to zero
  Matrix<real, Dynamic, 1> lambda(vlen, 1);
  for (int i = 0; i < qlen; ++i)
  {
    lambda(i * 4 + 0) = 1.f;
    lambda(i * 4 + 1) = 0.f;
    lambda(i * 4 + 2) = 0.f;
    lambda(i * 4 + 3) = 0.f;
  }

  // Solve the smallest Eigen value problem DL = EL
  // Where D is the self adjoint intrinsic dirac operator,
  // L is the similarity transformation, and E are the eigen values
  // Usually converges in 3 iterations or less
  lambda.normalize();
  for (int i = 0; i < back_substitutions + 1; ++i)
  {
    lambda = cg.solve(lambda.eval());
    lambda.normalize();
  }

  X.resize(qlen, 4);
  for (int i = 0; i < qlen; ++i)
  {
    X.row(i)(0) = lambda(i * 4 + 1);
    X.row(i)(1) = lambda(i * 4 + 2);
    X.row(i)(2) = lambda(i * 4 + 3);
    X.row(i)(3) = lambda(i * 4 + 0);
  }
}
