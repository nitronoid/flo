#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

__global__ void d_cotangent_laplacian_triplets(
  const thrust::device_ptr<const real3> di_vertices,
  const thrust::device_ptr<const int3> di_faces,
  const thrust::device_ptr<const real> di_face_area,
  const uint i_nfaces,
  thrust::device_ptr<int> do_I,
  thrust::device_ptr<int> do_J,
  thrust::device_ptr<real> do_V);

FLO_API cusp::coo_matrix<int, real, cusp::device_memory>
cotangent_laplacian(const thrust::device_ptr<const real3> di_vertices,
                    const thrust::device_ptr<const int3> di_faces,
                    const thrust::device_ptr<const real> di_face_area,
                    const int i_nverts,
                    const int i_nfaces);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN

