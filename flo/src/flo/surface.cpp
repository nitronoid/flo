#include "flo/surface.hpp"

FLO_NAMESPACE_BEGIN

std::size_t Surface::n_vertices()
{
  return vertices.size();
}

std::size_t Surface::n_faces()
{
  return faces.size();
}

FLO_NAMESPACE_END

