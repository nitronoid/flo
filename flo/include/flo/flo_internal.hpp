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

#include "flo/function_ref.hpp"
#include <gsl/gsl-lite.hpp>
#include <Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4d)

#endif//FLO_INCLUDED