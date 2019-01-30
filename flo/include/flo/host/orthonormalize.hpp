#ifndef FLO_HOST_INCLUDED_ORTHONORMALIZE
#define FLO_HOST_INCLUDED_ORTHONORMALIZE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<real> orthonormalize(
    const gsl::span<const real> i_vectors, 
    const uint i_num_vectors, 
    nonstd::function_ref<
    real(const Eigen::Matrix<real, Eigen::Dynamic, 1>&, const Eigen::Matrix<real, Eigen::Dynamic, 1>&)> i_inner_product);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_ORTHONORMALIZE

