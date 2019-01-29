#include "flo/host/area.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/doublearea.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<double> area(
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
      [](double a) -> double { return 0.5 * a; });
  return face_area;
}

FLO_HOST_NAMESPACE_END
