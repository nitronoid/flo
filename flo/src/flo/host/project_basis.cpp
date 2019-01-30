#include "flo/host/project_basis.hpp"
#include "flo/host/flo_matrix_operation.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<real> project_basis(
      const gsl::span<const real> i_vector,
      const gsl::span<const real> i_basis, 
      const uint i_basis_cols, 
      nonstd::function_ref<
      real(const Matrix<real, Dynamic, 1>&, const Matrix<real, Dynamic, 1>&)> i_inner_product)
{
  const auto vlen = i_vector.size();
  std::vector<real> projected(i_vector.begin(), i_vector.end());
  auto kappa_dot = array_to_matrix(gsl::make_span(projected));
  Map<const Matrix<real, Dynamic, Dynamic>> u(i_basis.data(), vlen, i_basis_cols); 

  // Subtract the projected vector from the un-projected
  for (uint i = 0u; i < i_basis_cols; ++i)
  {
    kappa_dot -= i_inner_product(kappa_dot, u.col(i)) * u.col(i);
  }

  return projected;
}

FLO_HOST_NAMESPACE_END
