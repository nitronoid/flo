#include "flo/device/surface.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API std::size_t Surface::n_vertices() const noexcept
{
  return vertices.size();
}

FLO_API std::size_t Surface::n_faces() const noexcept
{
  return faces.size();
}

FLO_API Surface make_surface(const ::flo::host::Surface& i_host_surface)
{
  // Device memory surface to return
  cusp::array1d<real3, cusp::device_memory> d_vertices(
    i_host_surface.n_vertices());
  cusp::array1d<int3, cusp::device_memory> d_faces(i_host_surface.n_faces());

  // Get raw pointers to the vertex and face data
  auto raw_vert_ptr = (flo::real3*)(&i_host_surface.vertices[0][0]);
  auto raw_face_ptr = (int3*)(&i_host_surface.faces[0][0]);

  // Copy the vertices and faces
  thrust::copy(raw_vert_ptr,
               raw_vert_ptr + i_host_surface.n_vertices(),
               d_vertices.data());
  thrust::copy(raw_face_ptr,
               raw_face_ptr + i_host_surface.n_faces(),
               d_faces.data());

  return flo::device::Surface{std::move(d_vertices), std::move(d_faces)};
}

FLO_DEVICE_NAMESPACE_END

