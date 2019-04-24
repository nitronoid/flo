#include <iostream>
#include <numeric>
#include <unsupported/Eigen/SparseExtra>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/adjacency_list.h>
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/load_mesh.hpp"
#include "flo/host/vertex_normals.hpp"
#include "flo/host/vertex_mass.hpp"
#include "flo/host/valence.hpp"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/mean_curvature.hpp"
#include "flo/host/area.hpp"
#include "flo/host/similarity_xform.hpp"
#include "flo/host/spin_positions.hpp"
#include "flo/host/divergent_edges.hpp"

namespace
{
template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
matrix_offsets(const gsl::span<Eigen::Vector3i> i_faces,
               Eigen::SparseMatrix<T>& i_matrix)
{
  const int nfaces = i_faces.size();

  using namespace Eigen;
  Matrix<int, Dynamic, Dynamic, RowMajor> offsets(6, nfaces);

  auto get_offset = [&i_matrix](int r, int c) {
    const T* begin = i_matrix.valuePtr();
    return &i_matrix.coeffRef(r, c) - begin;
  };

  for (int i = 0; i < nfaces; ++i)
  {
    auto f = i_faces[i];
    offsets(0, i) = get_offset(f[1], f[0]);
    offsets(1, i) = get_offset(f[2], f[1]);
    offsets(2, i) = get_offset(f[0], f[2]);
    offsets(3, i) = get_offset(f[0], f[1]);
    offsets(4, i) = get_offset(f[1], f[2]);
    offsets(5, i) = get_offset(f[2], f[0]);
  }

  return offsets;
}

struct Adjacency
{
  std::vector<int> adjacency_keys;
  std::vector<int> adjacency;
  std::vector<int> valence;
  std::vector<int> cumulative_valence;
};

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& array2d)
{
  // Get the total number of elements in the 2d vector
  const int N = std::accumulate(
    array2d.begin(), array2d.end(), 0, [](int lhs, const std::vector<T>& rhs) {
      return lhs + rhs.size();
    });
  std::vector<T> array1d;
  array1d.reserve(N);
  for (const auto& row : array2d)
  {

    std::copy(row.begin(), row.end(), std::back_inserter(array1d));
  }
  return array1d;
}

Adjacency vertex_triangle_adjacency(const gsl::span<Eigen::Vector3i> i_faces)
{
  const auto F = flo::host::array_to_matrix(i_faces);
  const int nverts = F.maxCoeff() + 1;
  std::cout << "NV: " << nverts << '\n';
  Adjacency ret;

  Eigen::Matrix<int, Eigen::Dynamic, 1> VF;
  Eigen::Matrix<int, Eigen::Dynamic, 1> NI;
  igl::vertex_triangle_adjacency(F, nverts, VF, NI);
  ret.adjacency = flo::host::matrix_to_array(VF);
  ret.cumulative_valence = flo::host::matrix_to_array(NI);

  ret.valence.reserve(nverts);
  std::adjacent_difference(ret.cumulative_valence.begin() + 1,
                           ret.cumulative_valence.end(),
                           std::back_inserter(ret.valence));

  ret.adjacency_keys.reserve(ret.cumulative_valence.back());
  for (int i = 0; i < nverts; ++i)
  {
    const int v = ret.valence[i];
    for (int j = 0; j < v; ++j)
    {
      ret.adjacency_keys.push_back(i);
    }
  }

  return ret;
}

Adjacency vertex_vertex_adjacency(const gsl::span<Eigen::Vector3i> i_faces)
{
  const auto Fm = flo::host::array_to_matrix(i_faces);
  Eigen::MatrixXi F = Fm;

  const int nverts = F.maxCoeff() + 1;
  std::cout << "NV: " << nverts << '\n';
  Adjacency ret;

  std::vector<std::vector<int>> A;
  igl::adjacency_list(F, A, false);
  std::copy(A[0].begin(), A[0].end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout<<"\n";
  ret.adjacency = flatten(A);

  ret.valence.reserve(nverts);
  std::transform(A.begin(),
                 A.end(),
                 std::back_inserter(ret.valence),
                 [](const std::vector<int>& vec) -> int { return vec.size(); });

  ret.cumulative_valence.reserve(nverts + 1);
  ret.cumulative_valence.push_back(0);
  std::partial_sum(ret.valence.begin(),
                   ret.valence.end(),
                   std::back_inserter(ret.cumulative_valence));

  ret.adjacency_keys.reserve(ret.cumulative_valence.back());
  for (int i = 0; i < nverts; ++i)
  {
    const int v = ret.valence[i];
    for (int j = 0; j < v; ++j)
    {
      ret.adjacency_keys.push_back(i);
    }
  }

  return ret;
}
}  // namespace

int main()
{
  const std::string name = "spot";
  const std::string matrix_prefix = "matrices/" + name + "/";
  // We'll only write results from the host API in the application
  using namespace flo;
  using namespace flo::host;
  auto surf = load_mesh((name + ".obj").c_str());

  auto vta = vertex_triangle_adjacency(surf.faces);
  const auto VTAK = array_to_matrix(gsl::make_span(vta.adjacency_keys));
  const auto VTA = array_to_matrix(gsl::make_span(vta.adjacency));
  const auto VTV = array_to_matrix(gsl::make_span(vta.valence));
  const auto VTCV = array_to_matrix(gsl::make_span(vta.cumulative_valence));
  Eigen::saveMarketVector(
    VTAK, matrix_prefix + "vertex_triangle_adjacency/adjacency_keys.mtx");
  Eigen::saveMarketVector(
    VTA, matrix_prefix + "vertex_triangle_adjacency/adjacency.mtx");
  Eigen::saveMarketVector(
    VTV, matrix_prefix + "vertex_triangle_adjacency/valence.mtx");
  Eigen::saveMarketVector(
    VTCV, matrix_prefix + "vertex_triangle_adjacency/cumulative_valence.mtx");

  auto vva = vertex_vertex_adjacency(surf.faces);
  const auto VVAK = array_to_matrix(gsl::make_span(vva.adjacency_keys));
  const auto VVA = array_to_matrix(gsl::make_span(vva.adjacency));
  const auto VVV = array_to_matrix(gsl::make_span(vva.valence));
  const auto VVCV = array_to_matrix(gsl::make_span(vva.cumulative_valence));
  Eigen::saveMarketVector(
    VVAK, matrix_prefix + "vertex_vertex_adjacency/adjacency_keys.mtx");
  Eigen::saveMarketVector(
    VVA, matrix_prefix + "vertex_vertex_adjacency/adjacency.mtx");
  Eigen::saveMarketVector(
    VVV, matrix_prefix + "vertex_vertex_adjacency/valence.mtx");
  Eigen::saveMarketVector(
    VVCV, matrix_prefix + "vertex_vertex_adjacency/cumulative_valence.mtx");

  // Calculate smooth vertex normals
  const auto normals = vertex_normals(surf.vertices, surf.faces);
  const auto N = array_to_matrix(gsl::make_span(normals));
  Eigen::saveMarket(N, matrix_prefix + "vertex_normals/vertex_normals.mtx");

  // Calculate the cotangent laplacian for our mesh
  auto L = cotangent_laplacian(surf.vertices, surf.faces);
  L.makeCompressed();
  Eigen::saveMarket(
    L, matrix_prefix + "cotangent_laplacian/cotangent_laplacian.mtx");
  auto QL = to_real_quaternion_matrix(L);
  QL.conservativeResize(QL.rows() - 4, QL.cols() - 4);

  // Calculate the adjacency matrix offsets
  const auto O = matrix_offsets(surf.faces, L);
  Eigen::saveMarket(O, matrix_prefix + "adjacency_matrix_offset/offsets.mtx");

  // Calculate the vertex masses for our mesh
  const auto mass = vertex_mass(surf.vertices, surf.faces);
  const auto M = array_to_matrix(gsl::make_span(mass));
  Eigen::saveMarketVector(M, matrix_prefix + "vertex_mass/vertex_mass.mtx");

  // Calculate the signed mean curvature based on our vertex normals
  const auto mc = signed_mean_curvature(surf.vertices, L, mass, normals);
  const auto MC = array_to_matrix(gsl::make_span(mc));
  Eigen::saveMarketVector(
    MC, matrix_prefix + "mean_curvature/signed_mean_curvature.mtx");

  // Calculate all face areas
  const auto face_area = area(surf.vertices, surf.faces);
  const auto FA = array_to_matrix(gsl::make_span(face_area));
  Eigen::saveMarketVector(FA, matrix_prefix + "face_area/face_area.mtx");

  std::vector<real> rho(surf.n_vertices(), 3.f);
  // Calculate the intrinsic dirac operator matrix
  const auto D =
    intrinsic_dirac(surf.vertices, surf.faces, vva.valence, face_area, rho);
  Eigen::saveMarket(D, matrix_prefix + "intrinsic_dirac/intrinsic_dirac.mtx");

  // Calculate the scaling and rotation for our spin transformation
  const auto lambda = similarity_xform(D);
  const auto LAM = array_to_matrix(gsl::make_span(lambda));
  Eigen::saveMarket(LAM,
                    matrix_prefix + "similarity_xform/lambda.mtx");

  // Calculate our transformed edges
  auto new_edges = divergent_edges(surf.vertices, surf.faces, lambda, L);
  const auto E = array_to_matrix(gsl::make_span(new_edges));
  Eigen::saveMarket(E, matrix_prefix + "divergent_edges/edges.mtx");

  // Remove the final edge to ensure we are compatible with the sliced laplacian
  new_edges.pop_back();

  // Solve the final vertex positions
  const auto new_positions = spin_positions(QL, new_edges);
  const auto P = array_to_matrix(gsl::make_span(new_positions));
  Eigen::saveMarket(P, matrix_prefix + "spin_positions/positions.mtx");

  return 0;
}