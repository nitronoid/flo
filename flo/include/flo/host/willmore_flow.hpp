#ifndef FLO_INCLUDED_WILLMORE_FLOW
#define FLO_INCLUDED_WILLMORE_FLOW

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Matrix<real, 3, 1>> willmore_flow(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces,
    nonstd::function_ref<
    void(gsl::span<real> x, const gsl::span<const real> dx)> i_integrator);

FLO_HOST_NAMESPACE_END

#endif//FLO_INCLUDED_WILLMORE_FLOW

