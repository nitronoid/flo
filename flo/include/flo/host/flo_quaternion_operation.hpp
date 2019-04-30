#ifndef FLO_HOST_INCLUDED_QUATERNION_OPERATION
#define FLO_HOST_INCLUDED_QUATERNION_OPERATION

#include "flo/flo_internal.hpp"

#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

/// @brief Computes the quaternion quaternion product
inline FLO_API Eigen::Matrix<real, 4, 1>
hammilton_product(const Eigen::Matrix<real, 4, 1>& i_rhs,
                  const Eigen::Matrix<real, 4, 1>& i_lhs);

/// @brief Computes the quaternion quaternion for pure imaginary quaternions
inline FLO_API Eigen::Matrix<real, 4, 1>
hammilton_product(const Eigen::Matrix<real, 3, 1>& i_rhs,
                  const Eigen::Matrix<real, 3, 1>& i_lhs);

/// @brief Computes the real matrix block for this quaternion
inline FLO_API Eigen::Matrix<real, 4, 4>
quat_to_block(const Eigen::Matrix<real, 4, 1>& i_quat);

/// @brief Computes the conjugate for this quaternion
inline FLO_API Eigen::Matrix<real, 4, 1>
conjugate(const Eigen::Matrix<real, 4, 1>& i_quat);

#include "flo_quaternion_operation.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_QUATERNION_OPERATION
