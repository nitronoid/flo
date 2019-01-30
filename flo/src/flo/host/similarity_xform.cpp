#include "flo/host/similarity_xform.hpp"
#include <Eigen/CholmodSupport>

using namespace Eigen;

FLO_HOST_NAMESPACE_BEGIN

FLO_API std::vector<Matrix<real, 4, 1>> similarity_xform(
    const SparseMatrix<real>& i_dirac_matrix)
{
	// Calculate the length of our matrix,
	// and hence the number of quaternions we should expect
  const uint vlen = i_dirac_matrix.cols();
  const uint qlen = vlen / 4u;

	// If double precsion, use Cholmod solver
#ifdef FLO_USE_DOUBLE_PRECISION
  CholmodSupernodalLLT<SparseMatrix<real>> cg;
#else
  // Cholmod not supported for single precision
  ConjugateGradient<SparseMatrix<real>, Lower, DiagonalPreconditioner<real>> cg;
  cg.setTolerance(1e-7);
#endif

  cg.compute(i_dirac_matrix);

	// Init every real part to 1, all imaginary parts to zero
  Matrix<real, Dynamic, 1> lambda(vlen);
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
  std::vector<Matrix<real, 4, 1>> lambdaQ(qlen);
  for (uint i = 0u; i < qlen; ++i)
  {
		// W is last in a vector but first in a quaternion
    lambdaQ[i] = Matrix<real, 4, 1>(
        lambda(i*4u + 1u),
        lambda(i*4u + 2u),
        lambda(i*4u + 3u),
        lambda(i*4u + 0u));
  }
  return lambdaQ;
}

FLO_HOST_NAMESPACE_END
