#ifndef FLO_HOST_INCLUDED_PROJECT_BASIS
#define FLO_HOST_INCLUDED_PROJECT_BASIS

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

std::vector<double> project_basis(
      const gsl::span<const double> i_vectors,
      const gsl::span<const double> i_basis, 
      const uint i_basis_cols, 
      nonstd::function_ref<
      double(const Eigen::VectorXd&, const Eigen::VectorXd&)> i_inner_product);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_PROJECT_BASIS
