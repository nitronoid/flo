#include <iostream>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>

#include "flo/load_mesh.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"

using namespace Eigen;

void forward_euler(gsl::span<double> i_x,
                   const gsl::span<const double> i_dx,
                   const double i_t)
{
  std::transform(i_x.begin(), i_x.end(), i_dx.begin(), i_x.begin(),
                 [i_t](auto x, auto dx)
                 {
                   return x + dx * i_t;
                 });
}

int main()
{
  auto surf = flo::load_mesh("foo.obj");

  const auto integrator = [tao=0.95](auto x, const auto dx){
    return forward_euler(x, dx, tao);
  };

  for (int iter = 0; iter < 3; ++iter)
  {
    std::cout<<"Iteration: "<<iter<<'\n';
    surf.vertices = flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  }

  auto V = flo::host::array_to_matrix(gsl::make_span(surf.vertices));
  auto F = flo::host::array_to_matrix(gsl::make_span(surf.faces));
  igl::writeOBJ("bar.obj", V, F);

  return 0;
}
