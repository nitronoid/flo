template <typename DerivedV,
          typename DerivedF,
          typename DerivedVV,
          typename DerivedA,
          typename DerivedP>
FLO_API void intrinsic_dirac(const Eigen::MatrixBase<DerivedV>& V,
                             const Eigen::MatrixBase<DerivedF>& F,
                             const Eigen::MatrixBase<DerivedVV>& VV,
                             const Eigen::MatrixBase<DerivedA>& A,
                             const Eigen::MatrixBase<DerivedP>& P,
                             Eigen::SparseMatrix<real>& D)
{
  using namespace Eigen;
  // Find the max valence
  const int nnz = V.rows() * VV.maxCoeff() * 16;
  const int dim = V.rows() * 4;
  // Allocate for our Eigen problem matrix
  D.conservativeResize(dim, dim);
  D.reserve(Eigen::VectorXi::Constant(dim, VV.maxCoeff() * 4));

  // For every face
  for (int k = 0; k < F.rows(); ++k)
  {
    // Get a reference to the face vertex indices
    const auto& f = F.row(k);
    // Compute components of the matrix calculation for this face
    auto a = -1.f / (4.f * A(k));
    auto b = 1.f / 6.f;
    auto c = A(k) / 9.f;

    // Compute edge vectors as imaginary quaternions
    std::array<Matrix<real, 4, 1>, 3> edges;
    // opposing edge per vertex i.e. vertex one opposes edge 1->2
    edges[0].head<3>() = V.row(f(2)) - V.row(f(1));
    edges[1].head<3>() = V.row(f(0)) - V.row(f(2));
    edges[2].head<3>() = V.row(f(1)) - V.row(f(0));
    // Initialize real part to zero
    edges[0].w() = 0.f;
    edges[1].w() = 0.f;
    edges[2].w() = 0.f;

    // increment matrix entry for each ordered pair of vertices
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
        // W comes first in a quaternion but last in a vector
        Matrix<real, 4, 1> cur_quat(D.coeff(f[i] * 4 + 1, f[j] * 4),
                                    D.coeff(f[i] * 4 + 2, f[j] * 4),
                                    D.coeff(f[i] * 4 + 3, f[j] * 4),
                                    D.coeff(f[i] * 4 + 0, f[j] * 4));

        // Calculate the matrix component
        Matrix<real, 4, 1> q = a * hammilton_product(edges[i], edges[j]) +
                               b * (P(f(i)) * edges[j] - P(f(j)) * edges[i]);
        q.w() += P(f(i)) * P(f(j)) * c;
        // Sum it with any existing value
        cur_quat += q;

        // Write it back into our matrix
        auto block = quat_to_block(cur_quat);
        insert_block_sparse(block, D, f(i), f(j));
      }
  }
}

