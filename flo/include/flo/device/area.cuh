#ifndef FLO_DEVICE_INCLUDED_AREA
#define FLO_DEVICE_INCLUDED_AREA

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API thrust::device_vector<real> area(
    const thrust::device_ptr<const real3> i_vertices,
    const thrust::device_ptr<const int3> i_faces,
    const uint i_nfaces);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_AREA

