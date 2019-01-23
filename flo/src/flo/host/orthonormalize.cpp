#include "flo/host/orthonormalize.hpp"

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

std::vector<double> orthonormalize(
    const gsl::span<const double> i_vectors, 
    const uint i_num_vectors, 
    nonstd::function_ref<
    double(const VectorXd&, const VectorXd&)> i_inner_product)
{
  // Normalize is defined using a self inner product
  auto normalize = [&](const auto& x) {
    return x.array() / std::sqrt(i_inner_product(x, x));
  };
  const auto vlen = i_vectors.size() / i_num_vectors;

  // Map our input vectors as a matrix
  Map<const MatrixXd> v(i_vectors.data(), vlen, i_num_vectors); 

  // Declare and allocate space for our final basis matrix
  std::vector<double> basis(vlen * i_num_vectors);
  Map<MatrixXd> u(basis.data(), vlen, i_num_vectors); 

  // The first u0 is v0 normalized
  u.col(0) = normalize(v.col(0));
  // Gramm Schmit process
  for (uint i = 1u; i < i_num_vectors; ++i)
  {
    u.col(i) = v.col(i) - i_inner_product(v.col(i), u.col(0)) * u.col(0);
    for (uint k = 1u; k < i; ++k)
    {
      u.col(i) -= i_inner_product(u.col(i), u.col(k)) * u.col(k);
    }
    u.col(i) = normalize(u.col(i).eval());
  }

  return basis;
}


FLO_HOST_NAMESPACE_END


