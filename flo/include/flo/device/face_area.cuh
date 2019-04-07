#ifndef FLO_DEVICE_INCLUDED_AREA
#define FLO_DEVICE_INCLUDED_AREA

#include "flo/flo_internal.hpp"
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void
face_area(cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
          cusp::array2d<int, cusp::device_memory>::const_view di_faces,
          cusp::array1d<real, cusp::device_memory>::view do_face_area);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_AREA

