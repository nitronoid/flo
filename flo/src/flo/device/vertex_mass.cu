#include "flo/device/vertex_mass.cuh"
#include "flo/host/area.hpp"
//#define TMPGNUC __GNUC__
//#define __STRINGIFY(TEXT) #TEXT
//#define __WARNING(TEXT) __STRINGIFY(GCC warning TEXT)
//#define WARNING(VALUE) __WARNING(__STRINGIFY(N = VALUE))
//_Pragma (WARNING(TMPGNUC))
//#define __GNUC__ 6
//#include <cuda.h>
//#include <curand.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#undef TMPGNUC
//#define __GNUC__ TMPGNUC 

using namespace Eigen;

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__global__ void vertex_mass_impl(
    const gsl::span<const int> i_vertex_face_adjacency,
    const gsl::span<const int> i_vertex_face_valence,
    const gsl::span<const int> i_vertex_offset,
    const gsl::span<const double> i_face_area,
    gsl::span<double> io_mass)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  // one thread per mass
  if (tid >= io_mass.size()) return;

  const int valence = i_vertex_face_valence[tid];
  const int offset = i_vertex_offset[tid];

  double total_area = 0.;
  for (int i = 0; i < valence; ++i)
  {
    total_area += i_face_area[i_vertex_face_adjacency[offset + i]];
  }
  constexpr auto third = 1.f / 3.f;
  io_mass[tid] = total_area * third;
}
}

std::vector<double> vertex_mass(
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
