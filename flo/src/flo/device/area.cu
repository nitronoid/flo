#include "flo/device/area.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void
area(cusp::array1d<real, cusp::device_memory>::const_view di_vertices,
     cusp::array1d<int, cusp::device_memory>::const_view di_faces,
     cusp::array1d<real, cusp::device_memory>::view do_face_area)
{
  const int nfaces = di_faces.size() / 3;
  auto fit = thrust::make_zip_iterator(
    thrust::make_tuple(di_faces.begin() + nfaces * 0,
                       di_faces.begin() + nfaces * 1,
                       di_faces.begin() + nfaces * 2));

  const int nvertices = di_vertices.size() / 3;
  auto vit = thrust::make_zip_iterator(
    thrust::make_tuple(di_vertices.begin() + nvertices * 0,
                       di_vertices.begin() + nvertices * 1,
                       di_vertices.begin() + nvertices * 2));

  thrust::transform(fit,
                    fit + nfaces,
                    do_face_area.begin(),
                    [vit=vit] __device__(
                      const thrust::tuple<int, int, int>& face) {
                      real3 edge0, edge1;
                      {
                        const auto& v0 = *(vit + face.get<0>());
                        const auto& v1 = *(vit + face.get<1>());
                        const auto& v2 = *(vit + face.get<2>());

                        edge0.x = v1.get<0>() - v0.get<0>();
                        edge0.y = v1.get<1>() - v0.get<1>();
                        edge0.z = v1.get<2>() - v0.get<2>();

                        edge1.x = v2.get<0>() - v0.get<0>();
                        edge1.y = v2.get<1>() - v0.get<1>();
                        edge1.z = v2.get<2>() - v0.get<2>();
                      }
                      const real3 normal = cross(edge0, edge1);
                      return __fsqrt_rd(dot(normal, normal)) * 0.5;
                    });
}

FLO_DEVICE_NAMESPACE_END
