#include "flo/device/vertex_mass.cuh"
#include "flo/host/area.hpp"

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

FLO_API thrust::device_vector<double> vertex_mass(
    const thrust::device_ptr<double> di_face_area,
    const thrust::device_ptr<int> di_vertex_face_adjacency,
    const thrust::device_ptr<int> di_vertex_face_valence,
    const thrust::device_ptr<int> di_cumulative_valence,
    const uint i_nfaces,
    const uint i_nverts)
{
  thrust::device_vector<double> mass(i_nverts);

  size_t nthreads_per_block = 1024;
  size_t nblocks = i_nverts / nthreads_per_block + 1;
  vertex_mass_impl<<<nthreads_per_block, nblocks>>>(
      di_vertex_face_adjacency,
      di_vertex_face_valence,
      di_cumulative_valence,
      di_face_area,
      mass.data(),
      i_nverts);
  
  return mass;
}

FLO_DEVICE_NAMESPACE_END
