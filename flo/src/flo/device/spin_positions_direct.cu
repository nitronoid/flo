#include "flo/device/spin_positions_direct.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

namespace direct
{

namespace
{
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
               const real i_tolerance)
{
  cu_raii::sparse::Handle sparse_handle;
  cu_raii::solver::SolverSp solver;

  spin_positions(&sparse_handle,
                 &solver,
                 di_quaternion_laplacian,
                 di_edges,
                 do_vertices,
                 i_tolerance);
}

FLO_API void spin_positions(
  cu_raii::sparse::Handle* io_sparse_handle,
  cu_raii::solver::SolverSp* io_solver,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_quaternion_laplacian,
  cusp::array2d<real, cusp::device_memory>::const_view di_edges,
  cusp::array2d<real, cusp::device_memory>::view do_vertices,
  const real i_tolerance)
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

#if __CUDACC_VER_MAJOR__ < 10
  // Tell cusolver to use symamd reordering if we're compiling with cuda 9
  const int reorder = 2;
#else
  // Tell cusolver to use metis reordering if we're compiling with cuda 10
  const int reorder = 3;
#endif
  // cusolver will set this flag
  int singularity = -1;


  io_solver->status = cusolverSpScsrlsvchol(
    *io_solver,
    di_quaternion_laplacian.num_rows,
    di_quaternion_laplacian.num_entries,
    description_QL,
    di_quaternion_laplacian.values.begin().base().get(),
    row_offsets.data().get(),
    di_quaternion_laplacian.column_indices.begin().base().get(),
    b.begin().base().get(),
    i_tolerance,
    reorder,
    do_vertices.values.begin().base().get(),
    &singularity);
  io_solver->error_assert(__LINE__);
  if (singularity != -1)
    std::cout << "Singularity: " << singularity << '\n';

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
      xin_ptr, xin_ptr + do_vertices.num_cols, xout_ptr, quat_shfl{});
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


