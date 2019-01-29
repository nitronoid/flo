#include "flo/device/vertex_mass.cuh"
#include "flo/host/area.hpp"
#include <thrust/device_ptr.h>

using namespace Eigen;

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void vertex_mass_impl(
    const thrust::device_ptr<const int> di_vertex_face_adjacency,
    const thrust::device_ptr<const int> di_vertex_face_valence,
    const thrust::device_ptr<const int> di_vertex_offset,
    const thrust::device_ptr<const double> di_face_area,
    thrust::device_ptr<double> dio_mass,
    const uint i_n_verts)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  // one thread per mass
  if (tid >= i_n_verts) return;

  const int valence = di_vertex_face_valence[tid];
  const int offset = di_vertex_offset[tid];

  double total_area = 0.;
  for (int i = 0; i < valence; ++i)
  {
    total_area += di_face_area[di_vertex_face_adjacency[offset + i]];
  }
  constexpr auto third = 1.f / 3.f;
  dio_mass[tid] = total_area * third;
}
}

FLO_API std::vector<double> vertex_mass(
    const gsl::span<const Vector3d> i_vertices,
    const gsl::span<const Vector3i> i_faces)
{
  std::vector<double> mass(i_vertices.size());
  auto face_area = host::area(i_vertices, i_faces);

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

FLO_DEVICE_NAMESPACE_END
