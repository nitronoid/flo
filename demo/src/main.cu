#include <iostream>
#include <chrono>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/vertex_triangle_adjacency.h>

#include "flo/load_mesh.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/willmore_flow.hpp"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/host/vertex_triangle_adjacency.hpp"

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
  auto V = flo::host::array_to_matrix(gsl::make_span(surf.vertices));
  auto F = flo::host::array_to_matrix(gsl::make_span(surf.faces));

  //std::cout<<"V:\n"<<V<<'\n';
  //std::cout<<"F:\n"<<F<<'\n';

  //const auto integrator = [tao=0.95](auto x, const auto dx){
  //  return forward_euler(x, dx, tao);
  //};

  //for (int iter = 0; iter < 3; ++iter)
  //{
  //  std::cout<<"Iteration: "<<iter<<'\n';
  //  surf.vertices = flo::host::willmore_flow(surf.vertices, surf.faces, integrator);
  //}


  std::vector<int> adjacency(surf.n_faces() * 3);
  std::vector<int> valence(surf.n_vertices());
  std::vector<int> cumulative_valence(surf.n_vertices() + 1);
  auto time_begin = std::chrono::high_resolution_clock::now();
  flo::host::vertex_triangle_adjacency(
      surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);
  auto time_end = std::chrono::high_resolution_clock::now();
  auto VF = flo::host::array_to_matrix(gsl::make_span(adjacency));
  auto NI = flo::host::array_to_matrix(gsl::make_span(cumulative_valence));
  using namespace std::chrono;
  std::cout<<"Time taken: "<<duration_cast<nanoseconds>(time_end-time_begin).count()<<'\n';
  
  
  thrust::device_vector<int> d_face_verts(surf.n_faces() * 3);
  thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());
  thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);
  thrust::device_vector<int> d_valence(surf.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);

  time_begin = std::chrono::high_resolution_clock::now();
  flo::device::vertex_triangle_adjacency(
      d_face_verts.data(), 
      surf.n_faces(), 
      surf.n_vertices(), 
      d_adjacency.data(), 
      d_valence.data(), 
      d_cumulative_valence.data());
  cudaDeviceSynchronize();
  time_end = std::chrono::high_resolution_clock::now();

  using namespace std::chrono;
  std::cout<<"Time taken: "<<duration_cast<nanoseconds>(time_end-time_begin).count()<<'\n';
  
  //std::vector<int> h_adjacency(surf.n_faces() * 3);
  //thrust::copy(d_adjacency.begin(), d_adjacency.end(), h_adjacency.begin());
  //auto VF = flo::host::array_to_matrix(gsl::make_span(h_adjacency));

  //std::vector<int> h_cumulative_valence(surf.n_vertices() + 1);
  //thrust::copy(d_cumulative_valence.begin(), d_cumulative_valence.end(), h_cumulative_valence.begin());
  //auto NI = flo::host::array_to_matrix(gsl::make_span(h_cumulative_valence));

  //std::cout<<"VF:\n"<<VF<<'\n';
  //std::cout<<"NI:\n"<<NI<<'\n';


  igl::writeOBJ("bar.obj", V, F);

  return 0;
}
