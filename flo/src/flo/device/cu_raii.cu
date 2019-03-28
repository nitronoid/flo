#include "flo/device/cu_raii.cuh"
#include <iostream>
#include <array>

FLO_DEVICE_NAMESPACE_BEGIN

ScopedCuSolverSparse::ScopedCuSolverSparse()
{
  status = cusolverSpCreate(&handle);
}

ScopedCuSolverSparse::~ScopedCuSolverSparse()
{
  cusolverSpDestroy(handle);
}

ScopedCuSolverSparse::operator cusolverSpHandle_t() const noexcept
{
  return handle;
}

bool ScopedCuSolverSparse::error_check(int line) const noexcept
{
  if (status == CUSOLVER_STATUS_SUCCESS)
    return false;

  static constexpr std::array<const char*, 8> error_string = {
    "CUSOLVER_SUCCESS",
    "CUSOLVER_NOT_INITIALIZED",
    "CUSOLVER_ALLOC_FAILED",
    "CUSOLVER_INVALID_VALUE",
    "CUSOLVER_ARCH_MISMATCH",
    "CUSOLVER_EXECUTION_FAILED",
    "CUSOLVER_INTERNAL_ERROR",
    "CUSOLVER_MATRIX_TYPE_NOT_SUPPORTED"};

  std::cout << error_string[status];
  if (line != -1)
    std::cout << ", on line" << line;
  std::cout << '\n';
  return true;
}

void ScopedCuSolverSparse::error_assert(int line) const noexcept
{
  if (error_check(line))
    std::exit(1);
}

ScopedCuSparse::ScopedCuSparse()
{
  status = cusparseCreate(&handle);
}
ScopedCuSparse::~ScopedCuSparse()
{
  cusparseDestroy(handle);
}

ScopedCuSparse::operator cusparseHandle_t() const noexcept
{
  return handle;
}

bool ScopedCuSparse::error_check(int line) const noexcept
{
  if (status == CUSPARSE_STATUS_SUCCESS)
    return false;

  static constexpr std::array<const char*, 9> error_string = {
    "CUSPARSE_SUCCESS",
    "CUSPARSE_NOT_INITIALIZED",
    "CUSPARSE_ALLOC_FAILED",
    "CUSPARSE_INVALID_VALUE",
    "CUSPARSE_ARCH_MISMATCH",
    "CUSPARSE_MAPPING_ERROR",
    "CUSPARSE_EXECUTION_FAILED",
    "CUSPARSE_INTERNAL_ERROR",
    "CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED"};

  std::cout << error_string[status];
  if (line != -1)
    std::cout << ", on line" << line;
  std::cout << '\n';
  return true;
}

void ScopedCuSparse::error_assert(int line) const noexcept
{
  if (error_check(line))
    std::exit(1);
}

ScopedCuSparseMatrixDescription::ScopedCuSparseMatrixDescription()
{
  cusparseCreateMatDescr(&description);
}

ScopedCuSparseMatrixDescription::ScopedCuSparseMatrixDescription(
  cusparseStatus_t* io_status)
{
  *io_status = cusparseCreateMatDescr(&description);
}
ScopedCuSparseMatrixDescription::~ScopedCuSparseMatrixDescription()
{
  cusparseDestroyMatDescr(description);
}

ScopedCuSparseMatrixDescription::operator cusparseMatDescr_t() const noexcept
{
  return description;
}

FLO_DEVICE_NAMESPACE_END
