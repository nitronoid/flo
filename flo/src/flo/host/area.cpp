#include "flo/host/area.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <igl/doublearea.h>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<real> area(
    const gsl::span<const Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  auto V = array_to_matrix(i_vertices);
  auto F = array_to_matrix(i_faces);

  Matrix<real, Dynamic, Dynamic> double_area;
  igl::doublearea(V, F, double_area);
  std::vector<real> face_area(i_faces.size());
  std::transform(
      double_area.data(), double_area.data() + i_faces.size(), face_area.data(),
      [](real a) -> real { return 0.5 * a; });
  return face_area;
}

FLO_HOST_NAMESPACE_END
