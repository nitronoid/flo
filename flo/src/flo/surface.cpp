#include "flo/surface.hpp"

FLO_NAMESPACE_BEGIN

FLO_SHARED_API std::size_t Surface::n_vertices()
{
  return vertices.size();
}

FLO_SHARED_API std::size_t Surface::n_faces()
{
  return faces.size();
}

FLO_NAMESPACE_END

