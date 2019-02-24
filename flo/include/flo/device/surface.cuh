#ifndef FLO_DEVICE_INCLUDED_SURFACE
#define FLO_DEVICE_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include "flo/host/surface.hpp"
#include <thrust/device_vector.h>

FLO_DEVICE_NAMESPACE_BEGIN

struct Surface
{
  thrust::device_vector<real3> vertices;
  thrust::device_vector<int3> faces;

  FLO_API std::size_t n_vertices() const noexcept;
  FLO_API std::size_t n_faces() const noexcept;
};

FLO_API Surface make_surface(const ::flo::host::Surface& i_host_surface);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_SURFACE
