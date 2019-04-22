#include <iostream>
#include <numeric>
#include <igl/writeOBJ.h>
#include "flo/host/load_mesh.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"

using namespace Eigen;

template <typename T>
void forward_euler(gsl::span<T> i_x,
                   const gsl::span<const T> i_dx,
                   const double i_t)
{
  std::transform(
    i_x.begin(), i_x.end(), i_dx.begin(), i_x.begin(), [i_t](T x, T dx) {
      return x + dx * i_t;
    });
}

int main()
{
  auto surf = flo::host::load_mesh("foo.obj");

  flo::real tao = 0.95f;
  const auto integrator = [tao](gsl::span<flo::real> x,
                                const gsl::span<const flo::real> dx) {
    return forward_euler(x, dx, tao);
  };

  for (int iter = 0; iter < 3; ++iter)
  {
    std::cout << "Iteration: " << iter << '\n';
    surf.vertices =
      flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  }

  auto V = flo::host::array_to_matrix(gsl::make_span(surf.vertices));
  auto F = flo::host::array_to_matrix(gsl::make_span(surf.faces));

  igl::writeOBJ("bar.obj", V, F);

  return 0;
}
