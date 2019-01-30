#include "flo/host/flo_mesh_operation.hpp"
#include <numeric>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API void remove_mean(gsl::span<Matrix<real, 4, 1>> io_positions)
{
  Matrix<real, 4, 1> zeroQuat(0.f, 0.f, 0.f, 0.f); 
  auto sum = std::accumulate(
      io_positions.begin(), io_positions.end(), std::move(zeroQuat));
  auto average = sum / io_positions.size();
  for (auto& pos : io_positions) pos -= average;
}

FLO_API void normalize_positions(gsl::span<Matrix<real, 4, 1>> io_positions)
{
  real max_dist = 0.;
  for (const auto& qpos : io_positions) 
    max_dist = std::max(max_dist, qpos.squaredNorm());
  max_dist = std::sqrt(max_dist);
  auto coef = 1.f / std::move(max_dist);
  for (auto& qpos : io_positions) qpos *= coef;
}

FLO_HOST_NAMESPACE_END
