#ifndef FLO_HOST_INCLUDED_LOAD_MESH
#define FLO_HOST_INCLUDED_LOAD_MESH

#include "flo/flo_internal.hpp"
#include "surface.hpp"

FLO_HOST_NAMESPACE_BEGIN

FLO_API Surface load_mesh(gsl::czstring i_path);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_LOAD_MESH
