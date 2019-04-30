#ifndef FLO_DEVICE_INCLUDED_AREA
#define FLO_DEVICE_INCLUDED_AREA

#include "flo/flo_internal.hpp"
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @brief Computes the per face triangle area
///  @param di_vertices A row major matrix of vertex positions
///  @param di_faces A row major matrix of face vertex indices
///  @param do_face_area An array of face area values
FLO_API void
face_area(cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
          cusp::array2d<int, cusp::device_memory>::const_view di_faces,
          cusp::array1d<real, cusp::device_memory>::view do_face_area);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_AREA

