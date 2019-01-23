#ifndef FLO_HOST_INCLUDED_SIMILARITY_XFORM
#define FLO_HOST_INCLUDED_SIMILARITY_XFORM

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

std::vector<Eigen::Vector4d> similarity_xform(
    const Eigen::SparseMatrix<double>& i_dirac_matrix);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_SIMILARITY_XFORM
