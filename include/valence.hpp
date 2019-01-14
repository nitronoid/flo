#ifndef FLO_INCLUDED_VALENCE
#define FLO_INCLUDED_VALENCE

#include "flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_NAMESPACE_BEGIN

std::vector<int> valence(const gsl::span<const Eigen::Vector3i> i_faces);

FLO_NAMESPACE_END

#endif//FLO_INCLUDED_VALENCE
