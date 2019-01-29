#ifndef FLO_DEVICE_INCLUDED_AREA
#define FLO_DEVICE_INCLUDED_AREA

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API thrust::device_vector<double> area(
    const thrust::device_ptr<const Eigen::Vector3d> i_vertices,
    const thrust::device_ptr<const Eigen::Vector3i> i_faces,
    const uint i_nfaces);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_AREA

