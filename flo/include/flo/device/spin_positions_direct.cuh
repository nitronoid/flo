#ifndef FLO_DEVICE_INCLUDED_SPIN_POSITIONS_DIRECT
#define FLO_DEVICE_INCLUDED_SPIN_POSITIONS_DIRECT

#include "flo/flo_internal.hpp"
#include "flo/device/cu_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace direct
{
FLO_API void 
spin_positions(cusp::coo_matrix<int, real, cusp::device_memory>::const_view
                 di_quaternion_laplacian,
               cusp::array2d<real, cusp::device_memory>::const_view di_edges,
               cusp::array2d<real, cusp::device_memory>::view do_vertices,
               const real i_tolerance = 1e-7);

FLO_API void spin_positions(
  cu_raii::sparse::Handle* io_sparse_handle,
  cu_raii::solver::SolverSp* io_solver,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_quaternion_laplacian,
  cusp::array2d<real, cusp::device_memory>::const_view di_edges,
  cusp::array2d<real, cusp::device_memory>::view do_vertices,
  const real i_tolerance = 1e-7);

}

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_SPIN_POSITIONS_DIRECT


