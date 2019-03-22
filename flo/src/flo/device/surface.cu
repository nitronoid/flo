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
  // Device memory arrays to return
  cusp::array1d<flo::real3, cusp::device_memory> d_vertices(
    i_host_surface.n_vertices());
  cusp::array1d<int3, cusp::device_memory> d_faces(i_host_surface.n_faces());

  // Get raw pointers to the vertex and face data
  auto raw_vert_ptr = (const flo::real3*)(&i_host_surface.vertices[0][0]);
  auto raw_face_ptr = (const int3*)(&i_host_surface.faces[0][0]);

  cusp::array1d<flo::real3, cusp::host_memory>::const_view h_vertices(
    raw_vert_ptr, raw_vert_ptr + i_host_surface.n_vertices());
  cusp::array1d<int3, cusp::host_memory>::const_view h_faces(
    raw_face_ptr, raw_face_ptr + i_host_surface.n_faces());

  // Copy the vertices and faces
  d_vertices = h_vertices;
  d_faces = h_faces;

  return {d_vertices, d_faces};
}

FLO_API SurfaceViewMutable make_surface_view(::flo::device::Surface& i_surface)
{
  return {i_surface.vertices.data(),
          i_surface.faces.data(),
          i_surface.n_vertices(),
          i_surface.n_faces()};
}

FLO_API SurfaceViewImmutable
make_surface_view(const ::flo::device::Surface& i_surface)
{
  return {i_surface.vertices.data(),
          i_surface.faces.data(),
          i_surface.n_vertices(),
          i_surface.n_faces()};
}

FLO_DEVICE_NAMESPACE_END

