#include <iostream>
#include <numeric>
#include <igl/write_triangle_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/flo_quaternion_operation.hpp"
#include "flo/host/willmore_flow.hpp"
#include "flo/host/spin_positions.hpp"
#include "flo/device/surface.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/face_area.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/divergent_edges.cuh"
#include "flo/device/similarity_xform.cuh"
#include "flo/device/spin_positions.cuh"
#include <cusp/transpose.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <unsupported/Eigen/SparseExtra>

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
  ForwardEuler<flo::real> integrator(tao);

  flo::host::Surface h_surf;
  igl::read_triangle_mesh(in_name, h_surf.vertices, h_surf.faces);
  auto d_surf = flo::device::make_surface(h_surf);

  //----------------------------------------------------------------------------
  using namespace Eigen;
  // Calculate smooth vertex normals
  Matrix<flo::real, Dynamic, 3> N;
  igl::per_vertex_normals(h_surf.vertices,
                          h_surf.faces,
                          igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,
                          N);

  // Calculate the cotangent laplacian for our mesh
  SparseMatrix<flo::real> L;
  igl::cotmatrix(h_surf.vertices, h_surf.faces, L);
  L = (-L.eval());

  // Calculate the vertex masses for our mesh
  Matrix<flo::real, Dynamic, 1> M;
  flo::host::vertex_mass(h_surf.vertices, h_surf.faces, M);

  // Build our constraints {1, N.x, N.y, N.z}
  Matrix<flo::real, Dynamic, 4> constraints(N.rows(), 4);
  constraints.col(0) = Matrix<flo::real, Dynamic, 1>::Ones(N.rows());
  constraints.col(1) = N.col(0);
  constraints.col(2) = N.col(1);
  constraints.col(3) = N.col(2);

  // Declare an immersed inner-product using the mass matrix
  const auto ip = [&M](const Matrix<flo::real, Dynamic, 1>& x,
                       const Matrix<flo::real, Dynamic, 1>& y) -> flo::real {
    auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
    return single_mat(0, 0);
  };

  // Build a constraint basis using the Gram–Schmidt process
  Matrix<flo::real, Dynamic, Dynamic> U;
  flo::host::orthonormalize(constraints, ip, U);

  // Calculate the signed mean curvature based on our vertex normals
  Matrix<flo::real, Dynamic, 1> H;
  flo::host::signed_mean_curvature(h_surf.vertices, L, M, N, H);
  // Apply our flow direction to the mean curvature half density
  H *= -1.f;

  Matrix<flo::real, Dynamic, 1> HP = H;
  //// project the constraints on to our mean curvature
  flo::host::project_basis(HP, U, ip);
  // take a time step
  integrator(H, HP);
  //----------------------------------------------------------------------------
  // RHO
  cusp::array1d<flo::real, cusp::device_memory> d_rho(d_surf.n_vertices());
  thrust::copy_n(H.data(), d_surf.n_vertices(), d_rho.begin());

  // VERTEX VERTEX ADJACENCY
  const int nadjw = d_surf.n_faces() * 6;
  cusp::array1d<int, cusp::device_memory> d_adjacencyw(nadjw);
  cusp::array1d<int, cusp::device_memory> d_adjacency_keysw(nadjw);
  cusp::array1d<int, cusp::device_memory> d_valence(d_surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
    d_surf.n_vertices() + 1);
  d_cumulative_valence[0] = 0.f;
  const int nadjacency = flo::device::vertex_vertex_adjacency(
    d_surf.faces,
    d_adjacency_keysw,
    d_adjacencyw,
    d_valence,
    d_cumulative_valence.subarray(1, d_surf.n_vertices()));
  auto d_adjacency = d_adjacencyw.subarray(0, nadjacency);
  auto d_adjacency_keys = d_adjacency_keysw.subarray(0, nadjacency);

  // VERTEX TRIANGLE ADJACENCY
  cusp::array1d<int, cusp::device_memory> d_triangle_adjacency(
    d_surf.n_faces() * 3);
  cusp::array1d<int, cusp::device_memory> d_triangle_adjacency_keys(
    d_surf.n_faces() * 3);
  cusp::array1d<int, cusp::device_memory> d_triangle_valence(
    d_surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_triangle_cumulative_valence(
    d_surf.n_vertices() + 1);
  d_triangle_cumulative_valence[0] = 0.f;
  flo::device::vertex_triangle_adjacency(
    d_surf.faces,
    d_triangle_adjacency_keys,
    d_triangle_adjacency,
    d_triangle_valence,
    d_triangle_cumulative_valence.subarray(1, d_surf.n_vertices()));

  // ADJACENCY MATRIX OFFSETS
  cusp::array2d<int, cusp::device_memory> d_offsets(6, d_surf.n_faces());
  flo::device::adjacency_matrix_offset(
    d_surf.faces, d_adjacency, d_cumulative_valence, d_offsets);

  // COT MATRIX
  cusp::coo_matrix<int, flo::real, cusp::device_memory> d_L(
    d_surf.n_vertices(),
    d_surf.n_vertices(),
    d_cumulative_valence.back() + d_surf.n_vertices());
  // Allocate a dense 1 dimensional array to receive diagonal element indices
  cusp::array1d<int, cusp::device_memory> d_diagonals(d_surf.n_vertices());
  flo::device::cotangent_laplacian(d_surf.vertices,
                                   d_surf.faces,
                                   d_offsets,
                                   d_adjacency_keys,
                                   d_adjacency,
                                   d_cumulative_valence,
                                   d_diagonals,
                                   d_L);

  // FACE AREAS
  cusp::array1d<flo::real, cusp::device_memory> d_area(d_surf.n_faces());
  flo::device::face_area(d_surf.vertices, d_surf.faces, d_area);

  // INTRINSIC DIRAC MATRIX
  cusp::coo_matrix<int, flo::real4, cusp::device_memory> d_Dq(
    d_surf.n_vertices(),
    d_surf.n_vertices(),
    d_cumulative_valence.back() + d_surf.n_vertices());
  // Run our function
  flo::device::intrinsic_dirac(d_surf.vertices,
                               d_surf.faces,
                               d_area,
                               d_rho,
                               d_offsets,
                               d_adjacency_keys,
                               d_adjacency,
                               d_cumulative_valence,
                               d_triangle_adjacency_keys,
                               d_triangle_adjacency,
                               d_diagonals,
                               d_Dq);
  // Allocate our real matrix for solving
  cusp::coo_matrix<int, flo::real, cusp::device_memory> d_D(
    d_surf.n_vertices() * 4, d_surf.n_vertices() * 4, d_Dq.values.size() * 16);
  auto count = thrust::make_counting_iterator(1);
  thrust::transform(d_cumulative_valence.begin() + 1,
                    d_cumulative_valence.end(),
                    count,
                    d_cumulative_valence.begin() + 1,
                    thrust::plus<int>());
  // Transform our quaternion matrix to a real matrix
  flo::device::to_quaternion_matrix(
    d_Dq, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_D);

  std::cout << "d_D: " << d_D.num_rows << 'x' << d_D.num_cols << 'x'
            << d_D.num_entries << '\n';

  // SIMILARITY XFORM
  cusp::array2d<flo::real, cusp::device_memory> d_XT(4, d_surf.n_vertices());
  flo::device::similarity_xform(d_D, d_XT, 1e-7);
  cusp::array2d<flo::real, cusp::device_memory> d_X(d_XT.num_cols,
                                                    d_XT.num_rows);
  cusp::transpose(d_XT, d_X);

  // DIVERGENT EDGES
  cusp::array2d<flo::real, cusp::device_memory> d_E(4, d_surf.n_vertices());
  flo::device::divergent_edges(
    d_surf.vertices, d_surf.faces, d_X.values, d_L, d_E);

  // QUATERNION COT MATRIX
  cusp::coo_matrix<int, flo::real, cusp::device_memory> d_LQ(
    d_surf.n_vertices() * 4, d_surf.n_vertices() * 4, d_L.num_entries * 16);
  flo::device::to_real_quaternion_matrix(
    d_L, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_LQ);

  // SPIN POSITIONS
  cusp::array2d<flo::real, cusp::device_memory> d_vertices(
    4, d_surf.n_vertices(), 0.f);
  flo::device::spin_positions(d_LQ, d_E, d_vertices);

  thrust::copy_n(
    d_vertices.values.begin(), h_surf.n_vertices() * 3, h_surf.vertices.data());
  igl::write_triangle_mesh(out_name, h_surf.vertices, h_surf.faces);

  return 0;
}

