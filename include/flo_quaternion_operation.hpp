#ifndef FLO_INCLUDED_QUATERNION_OPERATION
#define FLO_INCLUDED_QUATERNION_OPERATION

#include "flo_internal.hpp"

#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

Eigen::Vector4d hammilton_product(
    const Eigen::Vector4d& i_rhs, const Eigen::Vector4d& i_lhs);

Eigen::Vector4d hammilton_product(
    const Eigen::Vector3d& i_rhs, const Eigen::Vector3d& i_lhs);

Eigen::Matrix4d quat_to_block(const Eigen::Vector4d& i_quat);

Eigen::Vector4d conjugate(const Eigen::Vector4d& i_quat);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_QUATERNION_OPERATION
