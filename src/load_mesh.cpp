#include "flo_internal.hpp"
#include "flo_matrix_operation.hpp"
#include "load_mesh.hpp"
#include <Eigen/Dense>
#include <igl/readOBJ.h>

using namespace Eigen;

FLO_NAMESPACE_BEGIN

Surface load_mesh(gsl::czstring i_path)
{
  Matrix<double, Dynamic, 3> V;
  Matrix<int, Dynamic, 3> F;
  igl::readOBJ(i_path, V, F);

  // Convert our eigen matrices to std vectors
  auto vertices = matrix_to_array(V);
  auto faces = matrix_to_array(F);

  // Return our arrays, with matrix masks
  return Surface{std::move(vertices), std::move(faces)};
}

FLO_NAMESPACE_END
