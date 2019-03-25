#include "flo/device/similarity_xform.cuh"
#include <thrust/tabulate.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/diagonal.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <chrono>
#include <cusp/permutation_matrix.h>
#include <cusp/graph/symmetric_rcm.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac_matrix,
  cusp::array1d<real, cusp::device_memory>::view do_xform)
{
  using namespace std::chrono;
  thrust::tabulate(do_xform.begin(), do_xform.end(), [] __device__(int x) {
    // When x is a multiple of 4, return one
    return !(x & 3);
  });

  // Allocate permutation matrix P
  cusp::permutation_matrix<int, cusp::device_memory> P(
    di_dirac_matrix.num_rows);
  // Construct symmetric RCM permutation matrix on the device
  auto t1 = high_resolution_clock::now();
  cusp::graph::symmetric_rcm(di_dirac_matrix, P);
  cusp::coo_matrix<int, real, cusp::device_memory> PA(di_dirac_matrix.num_rows, di_dirac_matrix.num_rows, di_dirac_matrix.values.size()); 
  PA = di_dirac_matrix;
  //P.symmetric_permute(PA);
  auto t2 = high_resolution_clock::now();
  std::cout << "Permute execution time: "
            << duration_cast<nanoseconds>(t2 - t1).count() << '\n';



  cusp::precond::diagonal<real, cusp::device_memory> M(PA);

  for (int i = 0; i < 3; ++i)
  {
    // Set-up stopping criteria
    cusp::monitor<real> monitor(do_xform, 8000, 1e-2);
    auto t1 = high_resolution_clock::now();
    // Conjugate gradient solve
    cusp::krylov::cg(PA, do_xform, do_xform, monitor, M);
    auto t2 = high_resolution_clock::now();
    std::cout << "Solve execution time: "
              << duration_cast<nanoseconds>(t2 - t1).count() << '\n';
    // Normalize the result by first finding the norm of the result
    auto norm =
      thrust::transform_reduce(do_xform.begin(),
                               do_xform.end(),
                               [] __device__(real x) -> real { return x * x; },
                               0.f,
                               thrust::plus<real>());
    // We then divide through by the sqrt of the norm
    thrust::transform(do_xform.begin(),
                      do_xform.end(),
                      do_xform.begin(),
                      [inv_len = 1.f / std::sqrt(norm)] __device__(real x) {
                        return x * inv_len;
                      });
    // report solver results
    if (monitor.converged())
    {
      std::cout << "Solver converged to " << monitor.relative_tolerance()
                << " relative tolerance";
      std::cout << " after " << monitor.iteration_count() << " iterations\n";
    }
    else
    {
      std::cout << "Solver reached iteration limit "
                << monitor.iteration_limit() << " before converging";
      std::cout << " to " << monitor.relative_tolerance()
                << " relative tolerance\n";
    }
  }

  // Re-arrange result to place W last
  thrust::scatter(
    do_xform.begin(),
    do_xform.end(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                    [] __device__(int x) {
                                      // Shuffle everything down by one, but if
                                      // you were an X component, add 4 to place
                                      // it as a W component
                                      return (x - 1) + !(x & 3) * 4;
                                    }),
    do_xform.begin());
}

FLO_DEVICE_NAMESPACE_END
