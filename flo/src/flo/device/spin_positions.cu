#include "flo/device/spin_positions.cuh"
#include <cusp/transpose.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
struct TupleNorm
{
  using Tup4 = thrust::tuple<real, real, real, real>;

  real operator()(const Tup4& vec) const
  {
    return vec.get<0>() * vec.get<0>() + vec.get<1>() * vec.get<1>() +
           vec.get<2>() * vec.get<2>() + vec.get<3>() * vec.get<3>();
  }
};

struct CenterFunctor
{
  real operator()(const thrust::tuple<real, real>& tup) const
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
               cusp::array2d<real, cusp::device_memory>::const_view di_xform,
               cusp::array2d<real, cusp::device_memory>::view do_vertices,
               const real i_tolerance,
               const int i_iterations)
{
  cu_raii::sparse::Handle sparse_handle;
  cu_raii::solver::SolverSp solver;
  auto io_solver = &solver;
  auto io_sparse_handle = &sparse_handle;

  // Transpose our transforms in preparation for matrix solve
  cusp::array2d<real, cusp::device_memory> b;
  cusp::transpose(di_xform, b);

  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(
    di_quaternion_laplacian.num_rows + 1);
  cusp::indices_to_offsets(di_quaternion_laplacian.row_indices, row_offsets);

  // Get a cuSolver and cuSparse handle
  io_solver->error_assert(__LINE__);
  io_sparse_handle->error_assert(__LINE__);

  // Create a matrix description
  cu_raii::sparse::MatrixDescription description_QL(&io_sparse_handle->status);
  io_sparse_handle->error_assert(__LINE__);

  // Tell cuSparse what matrix to expect
  cusparseSetMatType(description_QL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(description_QL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(description_QL, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(description_QL, CUSPARSE_INDEX_BASE_ZERO);

  // Tell cusolver to use metis reordering
  const int reorder = 3;
  // cusolver will set this flag
  int singularity = 0;

  io_solver->status = cusolverSpScsrlsvchol(
    *io_solver,
    di_quaternion_laplacian.num_rows,
    di_quaternion_laplacian.num_entries,
    description_QL,
    di_quaternion_laplacian.values.begin().base().get(),
    row_offsets.data().get(),
    di_quaternion_laplacian.column_indices.begin().base().get(),
    b.values.begin().base().get(),
    i_tolerance,
    reorder,
    do_vertices.values.begin().base().get(),
    &singularity);
  io_solver->error_assert(__LINE__);
  if (singularity != -1)
    std::cout << "Singularity: " << singularity << '\n';

  auto count_it = thrust::make_counting_iterator(0);
  auto discard_it = thrust::make_discard_iterator();

  thrust::scatter(do_vertices.values.begin(),
                  do_vertices.values.end(),
                  thrust::make_transform_iterator(
                    count_it,
                    [w = do_vertices.num_cols] __device__(int i) {
                      // Transpose our index, and
                      // simultaneously shuffle in the order:
                      // x -> w
                      // y -> x
                      // z -> y
                      // w -> z
                      const int32_t x = (i + 3) & 3;
                      const int32_t y = i >> 2;
                      return x * w + y;
                    }),
                  do_vertices.values.begin());

  // Iterator that provides the row of a vertex element
  auto row_begin = thrust::make_transform_iterator(
    count_it, [w = do_vertices.num_cols] __device__(int i) { return i / w; });
  auto row_end = row_begin + do_vertices.num_cols;

  // Find the sum of each vertex element
  thrust::device_vector<real> sum(4);
  thrust::reduce_by_key(
    row_begin, row_end, do_vertices.values.begin(), discard_it, sum.begin());

  // On read divide by the number of vertices to find an average
  const real coeff = 1.f / do_vertices.num_cols;
  auto avg_it = thrust::make_transform_iterator(
    sum.begin(), [=] __host__ __device__(real s) { return s * coeff; });

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
                    [=] __device__(real v) { return v * rmax; });
}

FLO_DEVICE_NAMESPACE_END

