#ifndef FLO_DEVICE_INCLUDED_CU_RAII
#define FLO_DEVICE_INCLUDED_CU_RAII

#include "flo/flo_internal.hpp"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

FLO_DEVICE_NAMESPACE_BEGIN

struct ScopedCuStream
{
  cudaStream_t handle;
  cudaError_t status;

  ScopedCuStream();

  ~ScopedCuStream();

  operator cudaStream_t() const noexcept;

  void join() noexcept;
};

struct ScopedCuSolverSparse
{
  cusolverSpHandle_t handle;
  cusolverStatus_t status;

  ScopedCuSolverSparse();

  ~ScopedCuSolverSparse();

  operator cusolverSpHandle_t() const noexcept;

  bool error_check(int line = -1) const noexcept;

  void error_assert(int line = -1) const noexcept;
};

struct ScopedCuSparse
{
  cusparseHandle_t handle;
  cusparseStatus_t status;

  ScopedCuSparse();

  ~ScopedCuSparse();

  operator cusparseHandle_t() const noexcept;

  bool error_check(int line = -1) const noexcept;

  void error_assert(int line = -1) const noexcept;
};

struct ScopedCuSparseMatrixDescription
{
  cusparseMatDescr_t description;

  ScopedCuSparseMatrixDescription();

  ScopedCuSparseMatrixDescription(cusparseStatus_t* io_status);

  ~ScopedCuSparseMatrixDescription();

  operator cusparseMatDescr_t() const noexcept;
};

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_CU_RAII

