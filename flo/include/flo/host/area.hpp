#ifndef FLO_HOST_INCLUDED_AREA
#define FLO_HOST_INCLUDED_AREA

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<real> area(
    const gsl::span<const Eigen::Matrix<real, 3, 1>> i_vertices,
    const gsl::span<const Eigen::Vector3i> i_faces);

FLO_HOST_NAMESPACE_END

#endif//FLO_HOST_INCLUDED_AREA
