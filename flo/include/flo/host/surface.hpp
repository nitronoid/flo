#ifndef FLO_HOST_INCLUDED_SURFACE
#define FLO_HOST_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

/// @brief A Simple struct to pack our vertices and faces together
struct Surface
{
  Eigen::Matrix<real, Eigen::Dynamic, 3> vertices;
  Eigen::Matrix<int, Eigen::Dynamic, 3> faces;

  inline FLO_API std::size_t n_vertices() const noexcept
  {
    return vertices.rows();
  }
  inline FLO_API std::size_t n_faces() const noexcept
  {
    return faces.rows();
  }
};

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SURFACE
