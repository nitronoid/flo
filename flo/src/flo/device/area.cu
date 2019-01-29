#include "flo/device/area.cuh"

using namespace Eigen;

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API thrust::device_vector<double> area(
    const thrust::device_ptr<const Vector3d> i_vertices,
    const thrust::device_ptr<const Vector3i> i_faces,
    const uint i_nfaces)
{
  thrust::device_vector<double> face_area(i_nfaces);
  thrust::transform(
      i_faces,
      i_faces + i_nfaces,
      face_area.begin(),
      [i_vertices] __host__ __device__ (const Eigen::Vector3i& face)
      {
        return (Eigen::Vector3d(i_vertices[face[1]]) - Eigen::Vector3d(i_vertices[face[0]]))
        .cross((Eigen::Vector3d(i_vertices[face[2]]) - Eigen::Vector3d(i_vertices[face[0]])))
        .norm() * 0.5;
      });
  return face_area;
}

FLO_DEVICE_NAMESPACE_END
