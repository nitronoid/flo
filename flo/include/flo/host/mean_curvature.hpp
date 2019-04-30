#ifndef FLO_HOST_INCLUDED_MEAN_CURVATURE
#define FLO_HOST_INCLUDED_MEAN_CURVATURE

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/flo_matrix_operation.hpp"
#include <algorithm>

FLO_HOST_NAMESPACE_BEGIN

/// @breif Computes the mean curvature normal per vertex
//  @param V #Vx3 A column major matrix of vertex positions
//  @param L #Vx#V Sparse cotangent laplacian matrix
//  @param M #V A per vertex array of masses
//  @param HN #Vx3 A column major matrix containing the computed normals
template <typename DerivedV, typename DerivedM, typename DerivedHN>
FLO_API void mean_curvature_normal(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   Eigen::PlainObjectBase<DerivedHN>& HN);

/// @breif Computes the mean curvature per vertex
//  @param V #Vx3 A column major matrix of vertex positions
//  @param L #Vx#V Sparse cotangent laplacian matrix
//  @param M #V A per vertex array of masses
//  @param H #V An array containing the computed mean curvatures
template <typename DerivedV, typename DerivedM, typename DerivedH>
FLO_API void mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                            const Eigen::SparseMatrix<real>& L,
                            const Eigen::MatrixBase<DerivedM>& M,
                            Eigen::PlainObjectBase<DerivedH>& H);

/// @breif Computes the signed mean curvature per vertex
//  @param V #Vx3 A column major matrix of vertex positions
//  @param L #Vx#V Sparse cotangent laplacian matrix
//  @param N #Vx3 A column major matrix of per vertex normals
//  @param M #V A per vertex array of masses
//  @param H #V An array containing the computed mean curvatures
template <typename DerivedV,
          typename DerivedM,
          typename DerivedN,
          typename DerivedH>
FLO_API void signed_mean_curvature(const Eigen::MatrixBase<DerivedV>& V,
                                   const Eigen::SparseMatrix<real>& L,
                                   const Eigen::MatrixBase<DerivedM>& M,
                                   const Eigen::MatrixBase<DerivedN>& N,
                                   Eigen::PlainObjectBase<DerivedH>& H);

#include "mean_curvature.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_MEAN_CURVATURE

