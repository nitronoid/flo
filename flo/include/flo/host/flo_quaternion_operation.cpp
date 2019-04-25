FLO_API Eigen::Matrix<real, 4, 1>
hammilton_product(const Eigen::Matrix<real, 4, 1>& i_rhs,
                  const Eigen::Matrix<real, 4, 1>& i_lhs)
{
  using namespace Eigen;
  const auto a1 = i_rhs.w();
  const auto b1 = i_rhs.x();
  const auto c1 = i_rhs.y();
  const auto d1 = i_rhs.z();
  const auto a2 = i_lhs.w();
  const auto b2 = i_lhs.x();
  const auto c2 = i_lhs.y();
  const auto d2 = i_lhs.z();
  // W is last in a vector
  return Matrix<real, 4, 1>(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
                            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2);
}

FLO_API Eigen::Matrix<real, 4, 1>
hammilton_product(const Eigen::Matrix<real, 3, 1>& i_rhs,
                  const Eigen::Matrix<real, 3, 1>& i_lhs)
{
  using namespace Eigen;
  return hammilton_product(
    Matrix<real, 4, 1>{i_rhs.x(), i_rhs.y(), i_rhs.z(), 0.},
    Matrix<real, 4, 1>{i_lhs.x(), i_lhs.y(), i_lhs.z(), 0.});
}

FLO_API Eigen::Matrix<real, 4, 4>
quat_to_block(const Eigen::Matrix<real, 4, 1>& i_quat)
{
  using namespace Eigen;
  const auto a = i_quat.w();
  const auto b = i_quat.x();
  const auto c = i_quat.y();
  const auto d = i_quat.z();

  Matrix<real, 4, 4> block;
  block << a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a;
  return block;
}

FLO_API Eigen::Matrix<real, 4, 1>
conjugate(const Eigen::Matrix<real, 4, 1>& i_quat)
{
  using namespace Eigen;
  return Matrix<real, 4, 1>(-i_quat.x(), -i_quat.y(), -i_quat.z(), i_quat.w());
}
