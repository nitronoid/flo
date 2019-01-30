#ifndef FLO_HOST_INCLUDED_INTRINSIC_DIRAC
#define FLO_HOST_INCLUDED_INTRINSIC_DIRAC

#include "flo/flo_internal.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API Eigen::SparseMatrix<real> intrinsic_dirac(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices, 
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const int> i_valence,
    const gsl::span<const real> i_face_area,
    const gsl::span<const real> i_rho);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_INTRINSIC_DIRAC
