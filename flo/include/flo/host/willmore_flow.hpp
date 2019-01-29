#ifndef FLO_INCLUDED_WILLMORE_FLOW
#define FLO_INCLUDED_WILLMORE_FLOW

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Vector3d> willmore_flow(
    const gsl::span<const Eigen::Vector3d> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    nonstd::function_ref<
    void(gsl::span<double> x, const gsl::span<const double> dx)> i_integrator);

FLO_HOST_NAMESPACE_END

#endif//FLO_INCLUDED_WILLMORE_FLOW

