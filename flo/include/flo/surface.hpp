#ifndef FLO_INCLUDED_SURFACE
#define FLO_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

struct Surface
{
  std::vector<Eigen::Matrix<real, 3, 1>> vertices;
  std::vector<Eigen::Vector3i> faces;

  FLO_SHARED_API std::size_t n_vertices();
  FLO_SHARED_API std::size_t n_faces();
};

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_SURFACE
