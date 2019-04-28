#ifndef FLO_HOST_INCLUDED_SIMILARITY_XFORM
#define FLO_HOST_INCLUDED_SIMILARITY_XFORM

#include "flo/flo_internal.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "flo/host/similarity_xform.hpp"
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>



FLO_HOST_NAMESPACE_BEGIN

template <typename DerivedX>
FLO_API void similarity_xform(const Eigen::SparseMatrix<real>& D,
                              Eigen::PlainObjectBase<DerivedX>& X,
                              int back_substitutions = 0);

#include "similarity_xform.cpp"

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SIMILARITY_XFORM
