#include "flo/host/surface.hpp"

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::size_t Surface::n_vertices() const noexcept
{
  return vertices.size();
}

FLO_API std::size_t Surface::n_faces() const noexcept
{
  return faces.size();
}

FLO_API SurfaceViewMutable make_surface_view(::flo::host::Surface& i_surface)
{
  return SurfaceViewMutable(gsl::make_span(i_surface.vertices), gsl::make_span(i_surface.faces));
}

FLO_API SurfaceViewImmutable
make_surface_view(const ::flo::host::Surface& i_surface)
{
  return SurfaceViewImmutable(gsl::make_span(i_surface.vertices), gsl::make_span(i_surface.faces));
}

FLO_HOST_NAMESPACE_END

