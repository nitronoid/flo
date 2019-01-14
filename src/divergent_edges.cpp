#include "divergent_edges.hpp"
#include "flo_quaternion_operation.hpp"

using namespace Eigen;

FLO_NAMESPACE_BEGIN

std::vector<Vector4d> divergent_edges(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces,
    const gsl::span<const Vector4d> i_lambda,
    const SparseMatrix<double> i_cotangent_laplacian)
{
  std::vector<Vector4d> new_edges(i_lambda.size(), {0.f, 0.f, 0.f, 0.f});

  // For every face
  for (const auto& f : i_faces)
  {
    // For each edge in the face
    for (uint i = 0u; i < 3u; ++i)
    {
      uint a = f[(i+1u) % 3u];
      uint b = f[(i+2u) % 3u];
      if (a > b) std::swap(a, b);

      const auto& l1 = i_lambda[a];
      const auto& l2 = i_lambda[b];

      auto edge = i_vertices[a] - i_vertices[b];
      Vector4d e(edge[0], edge[1], edge[2], 0.f);

      constexpr auto third = 1.f / 3.f;
      constexpr auto sixth = 1.f / 6.f;

      auto et = hammilton_product(hammilton_product(third*conjugate(l1),e),l1) +
                hammilton_product(hammilton_product(sixth*conjugate(l1),e),l2) +
                hammilton_product(hammilton_product(sixth*conjugate(l2),e),l1) +
                hammilton_product(hammilton_product(third*conjugate(l2),e),l2) ;

      auto cot_alpha = i_cotangent_laplacian.coeff(a, b) * 0.5f;
      new_edges[a] -= et * cot_alpha;
      new_edges[b] += et * cot_alpha;
    }
  }

  //removeMean(omega);
  return new_edges;
}

FLO_NAMESPACE_END
