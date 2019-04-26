#include <iostream>
#include <numeric>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"
#include "flo/host/surface.hpp"

using namespace Eigen;

// flo::host::Surface load_ply_mesh(gsl::czstring i_path)
//{
//  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> V;
//  Eigen::Matrix<int, Eigen::Dynamic, 3> F;
//  igl::readPLY(i_path, V, F);
//
//  // Convert our eigen matrices to std vectors
//  auto vertices = flo::host::matrix_to_array(V);
//  auto faces = flo::host::matrix_to_array(F);
//
//  // Return our arrays, with matrix masks
//  return flo::host::Surface{std::move(vertices), std::move(faces)};
//}
template <typename T>
struct ForwardEuler
{
  T tao = 0.95f;

  ForwardEuler(T i_tao) : tao(std::move(i_tao))
  {
  }

  void operator()(Eigen::Matrix<T, Eigen::Dynamic, 1>& i_x,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& i_dx) const
  {
    i_x += i_dx * tao;
  }
};

int main()
{
  flo::host::Surface surf;
  igl::readOBJ("foo.obj", surf.vertices, surf.faces);

  ForwardEuler<flo::real> integrator(0.95f);

  for (int iter = 0; iter < 1; ++iter)
  {
    std::cout << "Iteration: " << iter << '\n';
    flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  }

  igl::writeOBJ("bar.obj", surf.vertices, surf.faces);

  return 0;
}
