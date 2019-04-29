#include <iostream>
#include <numeric>
#include <unsupported/Eigen/SparseExtra>
#include <igl/readPLY.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/per_vertex_normals.h>
#include <igl/doublearea.h>
#include <igl/cotmatrix.h>
#include "flo/host/surface.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/vertex_mass.hpp"
#include "flo/host/vertex_vertex_adjacency.hpp"
#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/mean_curvature.hpp"
#include "flo/host/similarity_xform.hpp"
#include "flo/host/spin_positions.hpp"
#include "flo/host/divergent_edges.hpp"
#include "flo/host/orthonormalize.hpp"
#include "flo/host/project_basis.hpp"
#include "flo/host/spin_xform.hpp"

namespace
{
template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
matrix_indices(const Eigen::MatrixXi& F,
               Eigen::SparseMatrix<T>& i_matrix,
               Eigen::VectorXi& i_diagonals)
{
  const int nfaces = F.rows();
  const int nvertices = F.maxCoeff() + 1;

  using namespace Eigen;
  Matrix<int, Dynamic, Dynamic, RowMajor> indices(6, nfaces);

  auto get_index = [&i_matrix](int r, int c) {
    const T* begin = i_matrix.valuePtr();
    return &i_matrix.coeffRef(r, c) - begin;
  };

  for (int i = 0; i < nfaces; ++i)
  {
    auto f = F.row(i);
    indices(0, i) = get_index(f[1], f[0]);
    indices(1, i) = get_index(f[2], f[1]);
    indices(2, i) = get_index(f[0], f[2]);
    indices(3, i) = get_index(f[0], f[1]);
    indices(4, i) = get_index(f[1], f[2]);
    indices(5, i) = get_index(f[2], f[0]);
  }

  i_diagonals.resize(nvertices);
  for (int i = 0; i < nvertices; ++i)
  {
    i_diagonals(i) = get_index(i, i);
  }

  return indices;
}

}  // namespace

int main(int argc, char* argv[])
{
  using namespace flo;
  using namespace flo::host;

  const std::string name = argv[1];
  const std::string matrix_prefix = "matrices/" + name + "/";
  // We'll only write results from the host API in the application
  flo::host::Surface surf;
  igl::readOBJ((name + ".obj").c_str(), surf.vertices, surf.faces);

  // Vertex triangle adjacency vectors
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTAK, VTA, VTV, VTCV;
  vertex_triangle_adjacency(surf.faces, VTAK, VTA, VTV, VTCV);
  Eigen::saveMarketVector(
    VTAK, matrix_prefix + "vertex_triangle_adjacency/adjacency_keys.mtx");
  Eigen::saveMarketVector(
    VTA, matrix_prefix + "vertex_triangle_adjacency/adjacency.mtx");
  Eigen::saveMarketVector(
    VTV, matrix_prefix + "vertex_triangle_adjacency/valence.mtx");
  Eigen::saveMarketVector(
    VTCV, matrix_prefix + "vertex_triangle_adjacency/cumulative_valence.mtx");

  // Vertex vertex adjacency vectors
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVAK, VVA, VVV, VVCV;
  vertex_vertex_adjacency(surf.faces, VVAK, VVA, VVV, VVCV);
  Eigen::saveMarketVector(
    VVAK, matrix_prefix + "vertex_vertex_adjacency/adjacency_keys.mtx");
  Eigen::saveMarketVector(
    VVA, matrix_prefix + "vertex_vertex_adjacency/adjacency.mtx");
  Eigen::saveMarketVector(
    VVV, matrix_prefix + "vertex_vertex_adjacency/valence.mtx");
  Eigen::saveMarketVector(
    VVCV, matrix_prefix + "vertex_vertex_adjacency/cumulative_valence.mtx");

  // Calculate smooth vertex normals
  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> N;
  igl::per_vertex_normals(
    surf.vertices, surf.faces, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
  Eigen::saveMarket(N, matrix_prefix + "vertex_normals/vertex_normals.mtx");

  // Calculate the cotangent laplacian for our mesh
  Eigen::SparseMatrix<flo::real> L;
  igl::cotmatrix(surf.vertices, surf.faces, L);
  L = -(L.eval());
  auto QL = to_real_quaternion_matrix(L);
  L.makeCompressed();
  Eigen::saveMarket(
    L, matrix_prefix + "cotangent_laplacian/cotangent_laplacian.mtx");
  Eigen::saveMarket(QL,
                    matrix_prefix +
                      "cotangent_laplacian/quaternion_cotangent_laplacian.mtx");

  Eigen::VectorXi diag;
  // Calculate the adjacency matrix indices
  const auto O = matrix_indices(surf.faces, L, diag);
  Eigen::saveMarket(O, matrix_prefix + "adjacency_matrix_indices/indices.mtx");
  Eigen::saveMarket(diag, matrix_prefix + "cotangent_laplacian/diagonals.mtx");

  // Calculate the vertex masses for our mesh
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> M;
  flo::host::vertex_mass(surf.vertices, surf.faces, M);
  Eigen::saveMarketVector(M, matrix_prefix + "vertex_mass/vertex_mass.mtx");

  // Build our constraints {1, N.x, N.y, N.z}
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> constraints(N.rows(), 4);
  constraints.col(0) =
    Eigen::Matrix<flo::real, Eigen::Dynamic, 1>::Ones(N.rows());
  constraints.col(1) = N.col(0);
  constraints.col(2) = N.col(1);
  constraints.col(3) = N.col(2);

  // Declare an immersed inner-product using the mass matrix
  const auto ip =
    [&M](const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& x,
         const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& y) -> flo::real {
    auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
    return single_mat(0, 0);
  };

  // Build a constraint basis using the Gram–Schmidt process
  Eigen::Matrix<flo::real, Eigen::Dynamic, Eigen::Dynamic> U;
  orthonormalize(constraints, ip, U);
  Eigen::saveMarket(U, matrix_prefix + "orthonormalize/basis.mtx");

  // Calculate the signed mean curvature based on our vertex normals
  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> HN;
  flo::host::mean_curvature_normal(surf.vertices, L, M, HN);
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> H;
  flo::host::mean_curvature(surf.vertices, L, M, H);
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> SH;
  flo::host::signed_mean_curvature(surf.vertices, L, M, N, SH);
  Eigen::saveMarket(HN,
                    matrix_prefix + "mean_curvature/mean_curvature_normal.mtx");
  Eigen::saveMarketVector(H,
                          matrix_prefix + "mean_curvature/mean_curvature.mtx");
  Eigen::saveMarketVector(
    SH, matrix_prefix + "mean_curvature/signed_mean_curvature.mtx");

  SH *= -1.f;
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> HP = SH;
  //// project the constraints on to our mean curvature
  project_basis(HP, U, ip);
  Eigen::saveMarketVector(
    HP, matrix_prefix + "project_basis/projected_mean_curvature.mtx");
  SH += 0.95f * HP;
  Eigen::saveMarketVector(SH, matrix_prefix + "project_basis/rho.mtx");

  // Calculate all face areas
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> FA;
  igl::doublearea(surf.vertices, surf.faces, FA);
  FA *= 0.5f;
  Eigen::saveMarketVector(FA, matrix_prefix + "face_area/face_area.mtx");

  // Calculate the intrinsic dirac operator matrix
  Eigen::SparseMatrix<flo::real> D;
  flo::host::intrinsic_dirac(surf.vertices, surf.faces, VVV, FA, SH, D);
  Eigen::saveMarket(D, matrix_prefix + "intrinsic_dirac/intrinsic_dirac.mtx");

  // Calculate the scaling and rotation for our spin transformation
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> X;
  flo::host::similarity_xform(D, X);
  Eigen::saveMarket(X, matrix_prefix + "similarity_xform/lambda.mtx");

  // Calculate our transformed edges
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> E;
  flo::host::divergent_edges(surf.vertices, surf.faces, X, L, E);
  Eigen::saveMarket(E, matrix_prefix + "divergent_edges/edges.mtx");

  // Solve the final vertex positions
  Eigen::Matrix<flo::real, Eigen::Dynamic, Eigen::Dynamic> V;
  flo::host::spin_positions(QL, E, V);
  Eigen::saveMarket(V, matrix_prefix + "spin_positions/positions.mtx");

  return 0;
}
