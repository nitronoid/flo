#ifndef FLO_HOST_INCLUDED_QUATERNION_OPERATION
#define FLO_HOST_INCLUDED_QUATERNION_OPERATION

#include "flo/flo_internal.hpp"

#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API Eigen::Vector4d hammilton_product(
    const Eigen::Vector4d& i_rhs, const Eigen::Vector4d& i_lhs);

FLO_API Eigen::Vector4d hammilton_product(
    const Eigen::Vector3d& i_rhs, const Eigen::Vector3d& i_lhs);

FLO_API Eigen::Matrix4d quat_to_block(const Eigen::Vector4d& i_quat);

FLO_API Eigen::Vector4d conjugate(const Eigen::Vector4d& i_quat);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_QUATERNION_OPERATION
