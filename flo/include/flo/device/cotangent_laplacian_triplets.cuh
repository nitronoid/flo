#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_TRIPLETS
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_TRIPLETS

#include "flo/flo_internal.hpp"
#include <thrust/device_ptr.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API cusp::coo_matrix<int, real, cusp::device_memory>
cotangent_laplacian(const thrust::device_ptr<const real3> di_vertices,
                    const thrust::device_ptr<const int3> di_faces,
                    const thrust::device_ptr<const real> di_face_area,
                    const int i_nverts,
                    const int i_nfaces,
                    const int i_total_valence);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_TRIPLETS
