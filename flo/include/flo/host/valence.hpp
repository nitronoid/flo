#ifndef FLO_HOST_INCLUDED_VALENCE
#define FLO_HOST_INCLUDED_VALENCE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<int> valence(const gsl::span<const Eigen::Vector3i> i_faces);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_VALENCE
