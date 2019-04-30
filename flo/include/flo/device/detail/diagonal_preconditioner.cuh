#ifndef FLO_DEVICE_INCLUDED_DETAIL_DIAGONAL_PRECONDITIONER
#define FLO_DEVICE_INCLUDED_DETAIL_DIAGONAL_PRECONDITIONER

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/linear_operator.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @brief A diagonal preconditioner for conjugate gradient solving
namespace detail
{
class DiagonalPreconditioner 
  : public cusp::linear_operator<flo::real, cusp::device_memory>
{
  using Parent = cusp::linear_operator<flo::real, cusp::device_memory>;
  cusp::array1d<flo::real, cusp::device_memory> diagonal_reciprocals;

public:
  DiagonalPreconditioner(
    cusp::coo_matrix<int, flo::real, cusp::device_memory>::const_view di_A)
    : diagonal_reciprocals(di_A.num_rows)
  {
    // extract the main diagonal
    thrust::fill(diagonal_reciprocals.begin(), diagonal_reciprocals.end(), 0.f);
    thrust::scatter_if(di_A.values.begin(), di_A.values.end(),
                       di_A.row_indices.begin(),
                       thrust::make_transform_iterator(thrust::make_zip_iterator(
                       thrust::make_tuple(di_A.row_indices.begin(),
                                          di_A.column_indices.begin())),
                           cusp::equal_pair_functor<int>()),
                       diagonal_reciprocals.begin());

    // invert the entries
    thrust::transform(diagonal_reciprocals.begin(),
                      diagonal_reciprocals.end(),
                      diagonal_reciprocals.begin(),
                      cusp::reciprocal_functor<flo::real>());
  }

  template <typename VectorType1, typename VectorType2>
  void operator()(const VectorType1& x, VectorType2& y) const
  {
    cusp::blas::xmy(diagonal_reciprocals, x, y);
  }
};
}

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_DETAIL_DIAGONAL_PRECONDITIONER
