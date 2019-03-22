#include "flo/device/area.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void
area(cusp::array1d<real3, cusp::device_memory>::const_view di_vertices,
     cusp::array1d<int3, cusp::device_memory>::const_view di_faces,
     cusp::array1d<real, cusp::device_memory>::view do_face_area)
{
  thrust::transform(di_faces.begin(),
                    di_faces.end(),
                    do_face_area.begin(),
                    [d_vertices = di_vertices.begin().base().get()] __device__(
                      const int3& face) {
                      real3 normal;
                      {
                        const real3 v0 = d_vertices[face.x];
                        const real3 v1 = d_vertices[face.y];
                        const real3 v2 = d_vertices[face.z];
                        normal = cross(v1 - v0, v2 - v0);
                      }
                      return __fsqrt_rd(dot(normal, normal)) * 0.5;
                    });
}

FLO_DEVICE_NAMESPACE_END
