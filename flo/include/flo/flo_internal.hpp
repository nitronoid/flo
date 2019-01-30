#ifndef FLO_INCLUDED

#define FLO_VERSION_MAJOR			0
#define FLO_VERSION_MINOR			1
#define FLO_VERSION_PATCH			0
#define FLO_VERSION					 10
#define FLO_VERSION_MESSAGE			"FLO: version 0.1.0"
#define FLO_INCLUDED FLO_VERSION

#define FLO_NAMESPACE flo
#define FLO_NAMESPACE_BEGIN namespace FLO_NAMESPACE {
#define FLO_NAMESPACE_END }

#define FLO_HOST_NAMESPACE_BEGIN FLO_NAMESPACE_BEGIN namespace host {
#define FLO_HOST_NAMESPACE_END }}

#define FLO_DEVICE_NAMESPACE_BEGIN FLO_NAMESPACE_BEGIN namespace device {
#define FLO_DEVICE_NAMESPACE_END }}

#ifndef __host__
#define __host__ 
#endif
#ifndef __device__
#define __device__ 
#endif

#define FLO_API __host__
#define FLO_DEVICE_ONLY_API __device__
#define FLO_SHARED_API FLO_API FLO_DEVICE_ONLY_API

// We don't want gsl in cuda (it wasn't implemented properly)
#define gsl_api 

#include "flo/function_ref.hpp"
#include <gsl/gsl-lite.hpp>
#include <Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4d)

FLO_NAMESPACE_BEGIN
#ifndef FLO_USE_DOUBLE_PRECISION
  using real = float;
  #ifdef __CUDACC__
  using real2 = float2;
  using real3 = float3;
  using real4 = float4;
  #endif
#else
  using real = double;
  #ifdef __CUDACC__
  using real2 = double2;
  using real3 = double3;
  using real4 = double4;
  #endif
#endif
FLO_NAMESPACE_END

#ifdef __CUDACC__
#include "flo/cuda_math_operation.cuh"
#endif

#endif//FLO_INCLUDED
