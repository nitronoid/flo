#include "similarity_xform.hpp"
#include <Eigen/CholmodSupport>

using namespace Eigen;

FLO_NAMESPACE_BEGIN

std::vector<Vector4d> similarity_xform(
    const SparseMatrix<double>& i_dirac_matrix)
{
	// Calculate the length of our matrix,
	// and hence the number of quaternions we should expect
  const uint vlen = i_dirac_matrix.cols();
  const uint qlen = vlen / 4u;
	// Cholmod solver
  CholmodSupernodalLLT<SparseMatrix<double>> cg;
  cg.compute(i_dirac_matrix);

	// Init every real part to 1, all imaginary parts to zero
  VectorXd lambda(vlen);
  lambda.setConstant(0.f);
  for (uint i = 0u; i < qlen; ++i) 
	{
		lambda(i*4u) = 1.f;
	}
	
	// Solve the smallest Eigen value problem DL = EL
	// Where D is the self adjoint intrinsic dirac operator,
	// L is the similarity transformation, and E are the eigen values
	// Usually converges in 3 iterations or less
  for (uint i = 0; i < 3; ++i)
  {
    lambda = cg.solve(lambda.eval());
  	lambda.normalize();
  }

	// Extract quaternions from our result matrix
  std::vector<Vector4d> lambdaQ(qlen);
  for (uint i = 0u; i < qlen; ++i)
  {
		// W is last in a vector but first in a quaternion
    lambdaQ[i] = Vector4d(
        lambda(i*4u + 1u),
        lambda(i*4u + 2u),
        lambda(i*4u + 3u),
        lambda(i*4u + 0u));
  }
  return lambdaQ;
}

FLO_NAMESPACE_END
