#include "flo/host/spin_positions.hpp"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/flo_mesh_operation.hpp"
#include <Eigen/CholmodSupport>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Matrix<real, 3, 1>> spin_positions(
    const SparseMatrix<real>& i_quaternion_laplacian, 
    const gsl::span<const Matrix<real, 4, 1>> i_edges)
{
  auto real_edge_matrix = to_quaternion_matrix(i_edges);
  // Solve for our new positions
	// If double precsion, use Cholmod solver
#ifdef FLO_USE_DOUBLE_PRECISION
  CholmodSupernodalLLT<SparseMatrix<real>> cg;
#else
  // Cholmod not supported for single precision
  ConjugateGradient<SparseMatrix<real>, Lower, DiagonalPreconditioner<real>> cg;
  cg.setTolerance(1e-7);
#endif
  cg.compute(i_quaternion_laplacian);
  Matrix<real, Dynamic, 1> solved_positions = cg.solve(real_edge_matrix.col(0));

  // Extract the positions as quaternions
  auto quaternion_positions = to_quaternion_vector(solved_positions);
  // Add a final position
  quaternion_positions.push_back({0.f, 0.f, 0.f, 0.f}); 

  // Normalize positions
  remove_mean(quaternion_positions);
  normalize_positions(quaternion_positions);

  std::vector<Matrix<real, 3, 1>> new_positions;
  new_positions.reserve(quaternion_positions.size());
  std::transform(
      std::make_move_iterator(quaternion_positions.begin()),
      std::make_move_iterator(quaternion_positions.end()),
      std::back_inserter(new_positions),
      [](Matrix<real, 4, 1>&& qpos) { 
        return Matrix<real, 3, 1>(std::move(qpos.head<3>())); 
        });

  return new_positions;
}

FLO_HOST_NAMESPACE_END
