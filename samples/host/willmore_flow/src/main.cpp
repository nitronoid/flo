#include <iostream>
#include <numeric>
#include <igl/write_triangle_mesh.h>
#include <igl/read_triangle_mesh.h>
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"
#include "flo/host/surface.hpp"

using namespace Eigen;

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

int main(int argc, char* argv[])
{
  // Command line arguments
  const std::string in_name = argv[1];
  const std::string out_name = argv[2];
  const int max_iter = std::stoi(argv[3]);
  const flo::real tao = std::stof(argv[4]);

  flo::host::Surface surf;
  igl::read_triangle_mesh(in_name, surf.vertices, surf.faces);

  ForwardEuler<flo::real> integrator(tao);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    std::cout << "Iteration: " << iter << '\n';
    flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  }

  igl::write_triangle_mesh(out_name, surf.vertices, surf.faces);

  return 0;
}
