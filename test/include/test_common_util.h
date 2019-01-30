#ifndef FLO_INCLUDED_TEST_COMMON_UTIL
#define FLO_INCLUDED_TEST_COMMON_UTIL

#include "flo/surface.hpp"

inline flo::Surface make_cube()
{
  std::vector<Eigen::Matrix<float, 3, 1>> vertices{
    {-0.5, -0.5,  0.5},
    { 0.5, -0.5,  0.5},
    {-0.5,  0.5,  0.5},
    { 0.5,  0.5,  0.5},
    {-0.5,  0.5, -0.5},
    { 0.5,  0.5, -0.5},
    {-0.5, -0.5, -0.5},
    { 0.5, -0.5, -0.5}
  };
  std::vector<Eigen::Vector3i> faces {
    {0, 1, 2},
    {2, 1, 3},
    {2, 3, 4},
    {4, 3, 5},
    {4, 5, 6},
    {6, 5, 7},
    {6, 7, 0},
    {0, 7, 1},
    {1, 7, 3},
    {3, 7, 5},
    {6, 0, 4},
    {4, 0, 2}
  };
  return {vertices, faces};
}

#endif//FLO_INCLUDED_TEST_COMMON_UTIL
