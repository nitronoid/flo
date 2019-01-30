#include "flo/flo_internal.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/load_mesh.hpp"
#include <Eigen/Dense>
#include <igl/readOBJ.h>

using namespace Eigen;

FLO_NAMESPACE_BEGIN

FLO_API Surface load_mesh(gsl::czstring i_path)
{
  Matrix<real, Dynamic, 3> V;
  Matrix<int, Dynamic, 3> F;
  igl::readOBJ(i_path, V, F);

  // Convert our eigen matrices to std vectors
  auto vertices = host::matrix_to_array(V);
  auto faces = host::matrix_to_array(F);

  // Return our arrays, with matrix masks
  return Surface{std::move(vertices), std::move(faces)};
}

FLO_NAMESPACE_END
