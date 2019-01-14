#include "intrinsic_dirac.hpp"
#include "flo_matrix_operation.hpp"
#include "flo_quaternion_operation.hpp"

using namespace Eigen;

FLO_NAMESPACE_BEGIN

SparseMatrix<double> intrinsic_dirac(
    const gsl::span<const Vector3d> i_vertices, 
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const int> i_valence,
    const gsl::span<const double> i_face_area,
    const gsl::span<const double> i_rho)
{
  // Find the max valence
  uint mv = *std::max_element(i_valence.begin(), i_valence.end());

  const uint vlen = i_vertices.size();
  // Allocate for our Eigen problem matrix
  SparseMatrix<double> D(vlen * 4u, vlen * 4u);
  D.reserve(Eigen::VectorXi::Constant(vlen*4u, mv));

  // For every face
  for (uint k = 0u; k < i_faces.size(); ++k)
  {
    // Get a reference to the face vertex indices
    const auto& f = i_faces[k];
    // Compute components of the matrix calculation for this face
    auto a = -1.f / (4.f*i_face_area[k]);
    auto b = 1.f / 6.f;
    auto c = i_face_area[k] / 9.f;

    // Compute edge vectors as imagnary quaternions
    std::array<Vector4d, 3> edges;
    edges[0].head<3>() = i_vertices[f[2]] - i_vertices[f[1]];
    edges[1].head<3>() = i_vertices[f[0]] - i_vertices[f[2]];
    edges[2].head<3>() = i_vertices[f[1]] - i_vertices[f[0]];
    // Init real part to zero
    edges[0].w() = 0.f;
    edges[1].w() = 0.f;
    edges[2].w() = 0.f;

    // increment matrix entry for each ordered pair of vertices
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      // W comes first in a quaternion but last in a vector
      Vector4d cur_quat(
          D.coeff(f[i] * 4 + 1, f[j] * 4),
          D.coeff(f[i] * 4 + 2, f[j] * 4),
          D.coeff(f[i] * 4 + 3, f[j] * 4),
          D.coeff(f[i] * 4 + 0, f[j] * 4));

      // Calculate the matrix component
      Vector4d q =
        a * hammilton_product(edges[i], edges[j]) +
        b * (i_rho[i] * edges[j] - i_rho[j] * edges[i]);
      q.w() += i_rho[i] * i_rho[j] * c;
      // Sum it with any exisiting value
      cur_quat += q;

      // Write it back into our matrix
      auto block = quat_to_block(cur_quat);
      insert_block_sparse(block, D, f[i], f[j]);
    }
  }
  return D;
}

FLO_NAMESPACE_END

