#ifndef FLO_INCLUDED_SIMILARITY_XFORM
#define FLO_INCLUDED_SIMILARITY_XFORM

#include "flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_NAMESPACE_BEGIN

std::vector<Eigen::Vector4d> similarity_xform(
    const Eigen::SparseMatrix<double>& i_dirac_matrix);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_SIMILARITY_XFORM
