#ifndef FLO_DEVICE_INCLUDED_CU_RAII
#define FLO_DEVICE_INCLUDED_CU_RAII

#include "flo/flo_internal.hpp"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace cu_raii
{
  struct Stream
  {
    cudaStream_t handle;
    cudaError_t status;

    Stream();
    ~Stream();

    operator cudaStream_t() const noexcept;
    void join() noexcept;
  };

  namespace solver
  {
  struct SolverSp
  {
    cusolverSpHandle_t handle;
    cusolverStatus_t status;

    SolverSp();
    ~SolverSp();

    operator cusolverSpHandle_t() const noexcept;
    bool error_check(int line = -1) const noexcept;
    void error_assert(int line = -1) const noexcept;
  };
  }  // namespace solver

  namespace sparse
  {
  struct Handle
  {
    cusparseHandle_t handle;
    cusparseStatus_t status;

    Handle();
    ~Handle();

    operator cusparseHandle_t() const noexcept;
    bool error_check(int line = -1) const noexcept;
    void error_assert(int line = -1) const noexcept;
  };

  struct MatrixDescription
  {
    cusparseMatDescr_t description;

    MatrixDescription();
    MatrixDescription(cusparseStatus_t* io_status);
    ~MatrixDescription();

    operator cusparseMatDescr_t() const noexcept;
  };
  }  // namespace sparse
}

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_CU_RAII

