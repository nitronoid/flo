#include "flo/host/vertex_mass.hpp"
#include "flo/host/area.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

std::vector<double> vertex_mass(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  std::vector<double> mass(i_vertices.size());
  auto face_area = area(i_vertices, i_faces);

  // For every face
  for (uint i = 0; i < i_faces.size(); ++i)
  {
    const auto& f = i_faces[i];
    constexpr auto third = 1.f / 3.f;
    auto thirdArea = face_area[i] * third;

    mass[f(0)] += thirdArea;
    mass[f(1)] += thirdArea;
    mass[f(2)] += thirdArea;
  }

  return mass;
}

FLO_HOST_NAMESPACE_END
