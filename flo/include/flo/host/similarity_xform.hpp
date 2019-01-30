#ifndef FLO_HOST_INCLUDED_SIMILARITY_XFORM
#define FLO_HOST_INCLUDED_SIMILARITY_XFORM

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Eigen::Matrix<real, 4, 1>> similarity_xform(
    const Eigen::SparseMatrix<real>& i_dirac_matrix);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_SIMILARITY_XFORM
