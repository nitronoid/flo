#include "area.hpp"
#include "flo_matrix_operation.hpp"
#include <igl/doublearea.h>

using namespace Eigen;

FLO_NAMESPACE_BEGIN

std::vector<double> area(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  auto V = array_to_matrix(i_vertices);
  auto F = array_to_matrix(i_faces);

  MatrixXd double_area;
  igl::doublearea(V, F, double_area);
  std::vector<double> face_area(i_faces.size());
  std::transform(
      double_area.data(), double_area.data() + i_faces.size(), face_area.data(),
      [](auto a) -> double { return 0.5 * a; });
  return face_area;
}

FLO_NAMESPACE_END
