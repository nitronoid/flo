#include "flo/host/flo_quaternion_operation.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

Vector4d hammilton_product(const Vector4d& i_rhs, const Vector4d& i_lhs)
{
  const auto a1 = i_rhs.w();
  const auto b1 = i_rhs.x();
  const auto c1 = i_rhs.y();
  const auto d1 = i_rhs.z();
  const auto a2 = i_lhs.w();
  const auto b2 = i_lhs.x();
  const auto c2 = i_lhs.y();
  const auto d2 = i_lhs.z();
  // W is last in a vector
  return Vector4d(
      a1*b2 + b1*a2 + c1*d2 - d1*c2,
      a1*c2 - b1*d2 + c1*a2 + d1*b2,
      a1*d2 + b1*c2 - c1*b2 + d1*a2,
      a1*a2 - b1*b2 - c1*c2 - d1*d2);
}

Vector4d hammilton_product(const Vector3d& i_rhs, const Vector3d& i_lhs)
{
  return hammilton_product(
      Vector4d{i_rhs.x(), i_rhs.y(), i_rhs.z(), 0.}, 
      Vector4d{i_lhs.x(), i_lhs.y(), i_lhs.z(), 0.});
}

Matrix4d quat_to_block(const Vector4d& i_quat)
{
  const auto a = i_quat.w();
  const auto b = i_quat.x();
  const auto c = i_quat.y();
  const auto d = i_quat.z();

  Matrix4d block;
  block <<  a, -b, -c, -d,
            b,  a, -d,  c,
            c,  d,  a, -b,
            d, -c,  b,  a;
  return block;
}

Vector4d conjugate(const Vector4d& i_quat)
{
  return Vector4d(-i_quat.x(), -i_quat.y(), -i_quat.z(), i_quat.w());
}

FLO_HOST_NAMESPACE_END

