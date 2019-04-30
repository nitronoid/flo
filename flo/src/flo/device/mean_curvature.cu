#include "flo/device/mean_curvature.cuh"
#include "flo/device/detail/unary_functional.cuh"
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <thrust/iterator/transform_iterator.h>


FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void mean_curvature_normal(
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view di_vertices,
  cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_cotangent_laplacian,
  cusp::array1d<real, cusp::device_memory>::const_view di_vertex_mass,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view do_mean_curvature_normals)
{
  //auto mit = thrust::make_transform_iterator(
  //    di_vertex_mass.begin(), [] __host__ __device__ (real x)
  //    {
  //      return -6.f * x;
  //    });

  //cusp::print(di_cotangent_laplacian);
  //cusp::print(di_vertices);
  ////cusp::multiply(
  ////    di_cotangent_laplacian,
  ////    di_vertices,
  ////    do_mean_curvature_normals);
  //cusp::print(do_mean_curvature_normals);

  //thrust::transform(do_mean_curvature_normals.values.begin(),
  //                  do_mean_curvature_normals.values.end(),
  //                  thrust::make_permutation_iterator(
  //                    mit, thrust::make_transform_iterator(
  //                      thrust::make_counting_iterator(0),
  //                      detail::unary_modulo<int>(di_vertices.num_cols))),
  //                  do_mean_curvature_normals.values.begin(),
  //                  thrust::divides<real>());


}

FLO_DEVICE_NAMESPACE_END
