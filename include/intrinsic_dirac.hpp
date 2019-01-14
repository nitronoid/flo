#ifndef FLO_INCLUDED_INTRINSIC_DIRAC
#define FLO_INCLUDED_INTRINSIC_DIRAC

#include "flo_internal.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

Eigen::SparseMatrix<double> intrinsic_dirac(
    const gsl::span<const Eigen::Vector3d> i_vertices, 
    const gsl::span<const Eigen::Vector3i> i_faces,
    const gsl::span<const int> i_valence,
    const gsl::span<const double> i_face_area,
    const gsl::span<const double> i_rho);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_INTRINSIC_DIRAC
