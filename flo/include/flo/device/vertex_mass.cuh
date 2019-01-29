#ifndef FLO_DEVICE_INCLUDED_VERTEX_MASS
#define FLO_DEVICE_INCLUDED_VERTEX_MASS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API thrust::device_vector<double> vertex_mass(
    const thrust::device_ptr<double> di_face_area,
    const thrust::device_ptr<int> di_vertex_face_adjacency,
    const thrust::device_ptr<int> di_vertex_face_valence,
    const thrust::device_ptr<int> di_cumulative_valence,
    const uint i_nfaces,
    const uint i_nverts);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_VERTEX_MASS

