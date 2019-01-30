#ifndef FLO_HOST_INCLUDED_PROJECT_BASIS
#define FLO_HOST_INCLUDED_PROJECT_BASIS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<real> project_basis(
      const gsl::span<const real> i_vectors,
      const gsl::span<const real> i_basis, 
      const uint i_basis_cols, 
      nonstd::function_ref<
      real(const Eigen::Matrix<real, Eigen::Dynamic, 1>&, const Eigen::Matrix<real, Eigen::Dynamic, 1>&)> i_inner_product);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_PROJECT_BASIS
