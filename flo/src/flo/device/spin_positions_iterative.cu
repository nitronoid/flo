#include "flo/device/spin_positions_direct.cuh"
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>
#include <cusp/precond/diagonal.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace iterative
{

namespace
{
class diagonal_precond : public cusp::linear_operator<flo::real, cusp::device_memory>
{
  using Parent = cusp::linear_operator<flo::real, cusp::device_memory>;
  cusp::array1d<flo::real, cusp::device_memory> diagonal_reciprocals;

public:
  diagonal_precond(cusp::coo_matrix<int, flo::real, cusp::device_memory>::const_view di_A)
    : diagonal_reciprocals(di_A.num_rows)
  {
    // extract the main diagonal
    thrust::fill(diagonal_reciprocals.begin(), diagonal_reciprocals.end(), 0.f);
    thrust::scatter_if(di_A.values.begin(), di_A.values.end(),
                       di_A.row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(thrust::make_tuple(
                               di_A.row_indices.begin(), di_A.column_indices.begin())),
                           cusp::equal_pair_functor<int>()),
                       diagonal_reciprocals.begin());

    // invert the entries
    thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                      diagonal_reciprocals.begin(), cusp::reciprocal_functor<flo::real>());
  }

  template <typename VectorType1, typename VectorType2>
  void operator()(const VectorType1& x, VectorType2& y) const
  {
    cusp::blas::xmy(diagonal_reciprocals, x, y);
  }
};
struct unary_divide
{
  unary_divide(int x) : denom(x)
  {
  }
  int denom;

  __host__ __device__ int operator()(int x) const
  {
    return x / denom;
  }
};

struct unary_multiply
{
  unary_multiply(real x) : coeff(x)
  {
  }
  real coeff;

  __host__ __device__ real operator()(real x) const
  {
    return x * coeff;
  }
};

struct quat_shuffle
{
  quat_shuffle(int x) : w(x)
  {
  }
  int w;

  __host__ __device__ int operator()(int i) const
  {
    // Shuffle in the order:
    // x -> w
    // y -> x
    // z -> y
    // w -> z
    const int32_t x = ((i / w) + 1) & 3;
    const int32_t y = i % w;
    return y * 4 + x;
  }
};

struct quat_shfl
{
  using tup4 = thrust::tuple<real, real, real, real>;

  __host__ __device__ tup4 operator()(real4 quat) const
  {
    return thrust::make_tuple(quat.y, quat.z, quat.w, quat.x);
  }
};

struct TupleNorm
{
  using Tup4 = thrust::tuple<real, real, real, real>;

  __host__ __device__ real operator()(const Tup4& vec) const
  {
    return vec.get<0>() * vec.get<0>() + vec.get<1>() * vec.get<1>() +
           vec.get<2>() * vec.get<2>() + vec.get<3>() * vec.get<3>();
  }
};

struct CenterFunctor
{
  __host__ __device__ real
  operator()(const thrust::tuple<real, real>& tup) const
  {
    return tup.get<0>() - tup.get<1>();
  }
};

template <typename ForwardIterator, typename ConstantIterator>
auto make_centered(ForwardIterator&& value_it, ConstantIterator&& const_it)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(std::forward<ForwardIterator>(value_it),
                         std::forward<ConstantIterator>(const_it))),
    CenterFunctor{}))
{
  return thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(std::forward<ForwardIterator>(value_it),
                         std::forward<ConstantIterator>(const_it))),
    CenterFunctor{});
}

}  // namespace

FLO_API void 
spin_positions(cusp::coo_matrix<int, real, cusp::device_memory>::const_view
                 di_quaternion_laplacian,
               cusp::array2d<real, cusp::device_memory>::const_view di_edges,
               cusp::array2d<real, cusp::device_memory>::view do_vertices,
               const real i_tolerance = 1e-7,
               const int i_max_convergence_iterations = 10000)
{
  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(
    di_quaternion_laplacian.num_rows + 1);
  cusp::indices_to_offsets(di_quaternion_laplacian.row_indices, row_offsets);

  cusp::array1d<real, cusp::device_memory> b(di_edges.num_entries);

  auto count_it = thrust::make_counting_iterator(0);
  auto shuffle_it =
    thrust::make_transform_iterator(count_it, quat_shuffle{di_edges.num_cols});
  thrust::scatter(
    di_edges.values.begin(), di_edges.values.end(), shuffle_it, b.begin());

  cusp::monitor<flo::real> monitor(
      b, i_max_convergence_iterations, i_tolerance, 0.f, false);
  diagonal_precond M(di_quaternion_laplacian);
    cusp::krylov::cg(di_quaternion_laplacian, do_vertices.values, b, monitor, M);
  {
    thrust::copy(
      do_vertices.values.begin(), do_vertices.values.end(), b.begin());
    auto xin_ptr =
      thrust::device_pointer_cast(reinterpret_cast<real4*>(b.data().get()));
    auto xout_ptr =
      thrust::make_zip_iterator(thrust::make_tuple(do_vertices.row(0).begin(),
                                                   do_vertices.row(1).begin(),
                                                   do_vertices.row(2).begin(),
                                                   do_vertices.row(3).begin()));

    thrust::transform(
      xin_ptr, xin_ptr + do_vertices.num_cols, xout_ptr, 
      [] __device__ (real4 quat)
      {
        return thrust::make_tuple(quat.y, quat.z, quat.w, quat.x);
      });

  }

  auto discard_it = thrust::make_discard_iterator();

  // Iterator that provides the row of a vertex element
  auto row_begin = thrust::make_transform_iterator(
    count_it, unary_divide(do_vertices.num_cols));
  auto row_end = row_begin + do_vertices.num_entries;

  // Find the sum of each vertex element
  thrust::device_vector<real> sum(4);
  thrust::reduce_by_key(
    row_begin, row_end, do_vertices.values.begin(), discard_it, sum.begin());

  // On read divide by the number of vertices to find an average
  auto avg_it = thrust::make_transform_iterator(
    sum.begin(), unary_multiply(1.f / do_vertices.num_cols));

  // Iterate over X, Y, Z and W elements of each vertex, and on read, first
  // subtract the average to center the vertex, then calculate the vector
  // squared norm using each element
  auto norm_begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(
      make_centered(do_vertices.row(0).begin(),
                    thrust::make_permutation_iterator(
                      avg_it, thrust::make_constant_iterator(0))),
      make_centered(do_vertices.row(1).begin(),
                    thrust::make_permutation_iterator(
                      avg_it, thrust::make_constant_iterator(1))),
      make_centered(do_vertices.row(2).begin(),
                    thrust::make_permutation_iterator(
                      avg_it, thrust::make_constant_iterator(2))),
      make_centered(do_vertices.row(3).begin(),
                    thrust::make_permutation_iterator(
                      avg_it, thrust::make_constant_iterator(3))))),
    TupleNorm{});
  auto norm_end = norm_begin + do_vertices.num_cols;

  // Find the reciprocal of the largest vertex norm
  const real rmax = 1.f / std::sqrt(*thrust::max_element(norm_begin, norm_end));
  // Scale all vertices by the reciprocal norm
  thrust::transform(do_vertices.values.begin(),
                    do_vertices.values.end(),
                    do_vertices.values.begin(),
                    unary_multiply(rmax));
}

}

FLO_DEVICE_NAMESPACE_END

