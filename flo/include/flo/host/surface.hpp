#ifndef FLO_HOST_INCLUDED_SURFACE
#define FLO_HOST_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

struct Surface
{
  std::vector<Eigen::Matrix<real, 3, 1>> vertices;
  std::vector<Eigen::Vector3i> faces;

  FLO_API std::size_t n_vertices() const noexcept;
  FLO_API std::size_t n_faces() const noexcept;
};

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SURFACE
