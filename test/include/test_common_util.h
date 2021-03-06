#ifndef FLO_INCLUDED_TEST_COMMON_UTIL
#define FLO_INCLUDED_TEST_COMMON_UTIL

#include "flo/host/surface.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include <unsupported/Eigen/SparseExtra>
#if defined(__CUDACC__)
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>

using DeviceVectorI = cusp::array1d<int, cusp::device_memory>;
using HostVectorI = cusp::array1d<int, cusp::host_memory>;
using DeviceVectorR = cusp::array1d<flo::real, cusp::device_memory>;
using HostVectorR = cusp::array1d<flo::real, cusp::host_memory>;

using DeviceDenseMatrixI = cusp::array2d<int, cusp::device_memory>;
using HostDenseMatrixI = cusp::array2d<int, cusp::host_memory>;
using DeviceDenseMatrixR = cusp::array2d<flo::real, cusp::device_memory>;
using HostDenseMatrixR = cusp::array2d<flo::real, cusp::host_memory>;

using DeviceSparseMatrixR =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using HostSparseMatrixR = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

using DeviceSparseMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using HostSparseMatrixQ = cusp::coo_matrix<int, flo::real4, cusp::host_memory>;

template <typename T>
cusp::array1d<T, cusp::host_memory> read_host_vector(std::string path)
{
  cusp::array1d<T, cusp::host_memory> ret;
  cusp::io::read_matrix_market_file(ret, path);
  return ret;
}

template <typename T>
cusp::array2d<T, cusp::host_memory> read_host_dense_matrix(std::string path)
{
  cusp::array2d<T, cusp::host_memory> ret;
  cusp::io::read_matrix_market_file(ret, path);
  return ret;
}

template <typename T>
cusp::coo_matrix<int, T, cusp::host_memory>
read_host_sparse_matrix(std::string path)
{
  cusp::coo_matrix<int, T, cusp::host_memory> ret;
  cusp::io::read_matrix_market_file(ret, path);
  return ret;
}

template <typename T>
cusp::array1d<T, cusp::device_memory> read_device_vector(std::string path)
{
  return read_host_vector<T>(path);
}

template <typename T>
cusp::array2d<T, cusp::device_memory> read_device_dense_matrix(std::string path)
{
  return read_host_dense_matrix<T>(path);
}

template <typename T>
cusp::coo_matrix<int, T, cusp::device_memory>
read_device_sparse_matrix(std::string path)
{
  return read_host_sparse_matrix<T>(path);
}

#else

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> read_vector(std::string path)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> ret;
  Eigen::loadMarketVector(ret, path);
  return ret;
}

template <typename T, int DIM = Eigen::Dynamic>
Eigen::Matrix<T, Eigen::Dynamic, DIM>
read_dense_matrix(std::string path)
{
  Eigen::SparseMatrix<T> temp;
  Eigen::loadMarket(temp, path);
  Eigen::Matrix<T, Eigen::Dynamic, DIM> ret = temp;
  return ret;
}

template <typename T>
Eigen::SparseMatrix<T> read_sparse_matrix(std::string path)
{
  Eigen::SparseMatrix<T> ret;
  Eigen::loadMarket(ret, path);
  return ret;
}

#endif

#endif  // FLO_INCLUDED_TEST_COMMON_UTIL
