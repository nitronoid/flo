#ifndef FLO_HOST_INCLUDED_QUATERNION_OPERATION
#define FLO_HOST_INCLUDED_QUATERNION_OPERATION

#include "flo/flo_internal.hpp"

#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API Eigen::Matrix<real, 4, 1> hammilton_product(
    const Eigen::Matrix<real, 4, 1>& i_rhs, const Eigen::Matrix<real, 4, 1>& i_lhs);

FLO_API Eigen::Matrix<real, 4, 1> hammilton_product(
    const Eigen::Matrix<real, 3, 1>& i_rhs, const Eigen::Matrix<real, 3, 1>& i_lhs);

FLO_API Eigen::Matrix<real, 4, 4> quat_to_block(const Eigen::Matrix<real, 4, 1>& i_quat);

FLO_API Eigen::Matrix<real, 4, 1> conjugate(const Eigen::Matrix<real, 4, 1>& i_quat);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_QUATERNION_OPERATION
