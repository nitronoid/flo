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

FLO_HOST_NAMESPACE_END

