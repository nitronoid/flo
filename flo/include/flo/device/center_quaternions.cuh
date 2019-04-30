#ifndef FLO_DEVICE_INCLUDED_CENTER_QUATERNIONS
#define FLO_DEVICE_INCLUDED_CENTER_QUATERNIONS

#include "flo/flo_internal.hpp"
#include <cusp/array2d.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void center_quaternions(
    cusp::array2d<flo::real, cusp::device_memory>::view dio_quats);

FLO_DEVICE_NAMESPACE_END

#endif // FLO_DEVICE_INCLUDED_CENTER_QUATERNIONS
