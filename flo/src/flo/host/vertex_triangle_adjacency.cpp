#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/vertex_triangle_adjacency.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

void vertex_triangle_adjacency(
    const gsl::span<const Vector3i> i_faces,
    const uint i_nverts,
    gsl::span<int> o_adjacency,
    gsl::span<int> o_valence,
    gsl::span<int> o_cumulative_valence)
{
  const auto F = array_to_matrix(i_faces);
  auto o_VF = array_to_matrix(o_adjacency);
  auto o_NI = array_to_matrix(o_cumulative_valence);

  VectorXi VF(i_faces.size()*3);
  VectorXi NI(i_nverts + 1);
  igl::vertex_triangle_adjacency(F, i_nverts, VF, NI);

  o_VF.array() = VF.array();
  o_NI.array() = NI.array();
  std::adjacent_difference(
      o_cumulative_valence.begin()+1, 
      o_cumulative_valence.end(), 
      o_valence.begin());
}

FLO_HOST_NAMESPACE_END
