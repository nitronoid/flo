#ifndef FLO_DEVICE_INCLUDED_VERTEX_MASS
#define FLO_DEVICE_INCLUDED_VERTEX_MASS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_DEVICE_NAMESPACE_BEGIN

//namespace impl
//{
//__global__ void d_parallelIntAdd(int* a, int* b, int* c);
//}

FLO_API std::vector<double> vertex_mass(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_VERTEX_MASS

