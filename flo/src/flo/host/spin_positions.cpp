#include "flo/host/spin_positions.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/flo_mesh_operation.hpp"
#include <Eigen/CholmodSupport>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

std::vector<Vector3d> spin_positions(
    const SparseMatrix<double>& i_quaternion_laplacian, 
    const gsl::span<const Vector4d> i_edges)
{
  auto real_edge_matrix = to_quaternion_matrix(i_edges);
  // Solve for our new positions
  CholmodSupernodalLLT<SparseMatrix<double>> cg;
  cg.compute(i_quaternion_laplacian);
  VectorXd solved_positions = cg.solve(real_edge_matrix.col(0));

  // Extract the positions as quaternions
  auto quaternion_positions = to_quaternion_vector(solved_positions);
  // Add a final position
  quaternion_positions.push_back({0.f, 0.f, 0.f, 0.f}); 

  // Normalize positions
  remove_mean(quaternion_positions);
  normalize_positions(quaternion_positions);

  std::vector<Vector3d> new_positions;
  new_positions.reserve(quaternion_positions.size());
  std::transform(
      std::make_move_iterator(quaternion_positions.begin()),
      std::make_move_iterator(quaternion_positions.end()),
      std::back_inserter(new_positions),
      [](auto&& qpos) { 
        using namespace std;
        return Vector3d(move(qpos.x()), move(qpos.y()), move(qpos.z())); 
        });

  return new_positions;
}

FLO_HOST_NAMESPACE_END
