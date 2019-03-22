#ifndef FLO_DEVICE_INCLUDED_SURFACE
#define FLO_DEVICE_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include "flo/host/surface.hpp"
#include <cusp/array1d.h>

FLO_DEVICE_NAMESPACE_BEGIN

struct Surface
{
  cusp::array1d<real3, cusp::device_memory> vertices;
  cusp::array1d<int3, cusp::device_memory> faces;

  FLO_API std::size_t n_vertices() const noexcept;
  FLO_API std::size_t n_faces() const noexcept;
};

template <bool CONST>
struct SurfaceView
{
private:
  template <bool ISCONST, typename T>
  using conditional_const_t =
    typename std::conditional<ISCONST,
                              typename std::add_const<T>::type,
                              typename std::remove_const<T>::type>::type;

public:
  SurfaceView(const SurfaceView&) = default;
  SurfaceView& operator=(const SurfaceView&) = default;
  SurfaceView(SurfaceView&&) = default;
  SurfaceView& operator=(SurfaceView&&) = default;
  ~SurfaceView() = default;

  // Only a non-const view can be constructed from non-const pointers
  template <typename DUMMY = void,
            typename = typename std::enable_if<!CONST, DUMMY>::type>
  SurfaceView(thrust::device_ptr<real3> i_vertices,
              thrust::device_ptr<int3> i_faces,
              std::size_t i_nvertices,
              std::size_t i_nfaces)
    : vertices(std::move(i_vertices))
    , faces(std::move(i_faces))
    , nvertices(std::move(i_nvertices))
    , nfaces(std::move(i_nfaces))
  {
  }

  // Create a const surface view from non-const or const pointers
  template <
    typename REAL3,
    typename INT3,
    typename DUMMY = void,
    typename = typename std::enable_if<
      CONST &&
      std::is_same<typename std::remove_const<REAL3>::type, real3>::value &&
      std::is_same<typename std::remove_const<INT3>::type, int3>::value,
                   DUMMY>::type>
      SurfaceView(thrust::device_ptr<REAL3> i_vertices,
                  thrust::device_ptr<INT3> i_faces,
                  std::size_t i_nvertices,
                  std::size_t i_nfaces) : vertices(std::move(i_vertices)),
    faces(std::move(i_faces)),
    nvertices(std::move(i_nvertices)),
    nfaces(std::move(i_nfaces))
  {
  }

  // Create a const surface view from a non-const
  template <typename DUMMY = void,
            typename = typename std::enable_if<CONST, DUMMY>::type>
  SurfaceView(const SurfaceView<false>& i_sv)
    : vertices(i_sv.vertices)
    , faces(i_sv.faces)
    , nvertices(i_sv.n_vertices())
    , nfaces(i_sv.n_faces())
  {
  }
  // Create a const surface view from a non-const
  template <typename DUMMY = void,
            typename = typename std::enable_if<CONST, DUMMY>::type>
  SurfaceView& operator=(const SurfaceView<false>& i_sv)
  {
    vertices = i_sv.vertices;
    faces = i_sv.faces;
    nvertices = i_sv.n_vertices();
    nfaces = i_sv.n_faces();
  }

  conditional_const_t<CONST,
                      thrust::device_ptr<conditional_const_t<CONST, real3>>>
    vertices;
  conditional_const_t<CONST,
                      thrust::device_ptr<conditional_const_t<CONST, int3>>>
    faces;

  FLO_API std::size_t n_vertices() const noexcept
  {
    return vertices.size();
  }

  FLO_API std::size_t n_faces() const noexcept
  {
    return faces.size();
  }

private:
  const std::size_t nvertices;
  const std::size_t nfaces;
};

using SurfaceViewMutable = SurfaceView<false>;
using SurfaceViewImmutable = SurfaceView<true>;

FLO_API Surface make_surface(const ::flo::host::Surface& i_host_surface);

FLO_API SurfaceViewMutable make_surface_view(::flo::device::Surface& i_surface);

FLO_API SurfaceViewImmutable
make_surface_view(const ::flo::device::Surface& i_surface);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_SURFACE
