#include "flo/device/area.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API thrust::device_vector<double> area(
    const thrust::device_ptr<const double3> i_vertices,
    const thrust::device_ptr<const int3> i_faces,
    const uint i_nfaces)
{
  thrust::device_vector<double> face_area(i_nfaces);
  thrust::transform(
      i_faces,
      i_faces + i_nfaces,
      face_area.begin(),
      [i_vertices] __host__ __device__ (const int3& face)
      {
        return length(cross(
            double3(i_vertices[face.y]) - double3(i_vertices[face.x]), 
            double3(i_vertices[face.z]) - double3(i_vertices[face.x]))) * 0.5;
      });
  return face_area;
}

FLO_DEVICE_NAMESPACE_END
