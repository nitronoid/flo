#include "test_common.h"

#include "flo/host/similarity_xform.hpp"

TEST(SimilarityXform, cube)
{
  auto cube = make_cube();

  Eigen::Matrix<flo::real, 32, 32> dense_D(32, 32);
  const flo::real o = 1.0;
  const flo::real z = 0.0;
  const flo::real f = 5.0;
  const flo::real m = 5.5;
  const flo::real h = 1.5;
  dense_D <<
   m, -z, -z, -z, -z, -o,  h,  h, -z, -h,  o, -h,  z,  z,  z,  z,  o, -z, -o, -o,  z,  z,  z,  z, -z,  h, -h, -z,  o,  o, -z,  o, 
   z,  m, -z,  z,  o, -z,  h, -h,  h, -z, -h, -o,  z,  z,  z,  z,  z,  o, -o,  o,  z,  z,  z,  z, -h, -z, -z,  h, -o,  o,  o,  z, 
   z,  z,  m, -z, -h, -h, -z, -o, -o,  h, -z, -h,  z,  z,  z,  z,  o,  o,  o, -z,  z,  z,  z,  z,  h,  z, -z,  h,  z, -o,  o,  o, 
   z, -z,  z,  m, -h,  h,  o, -z,  h,  o,  h, -z,  z,  z,  z,  z,  o, -o,  z,  o,  z,  z,  z,  z,  z, -h, -h, -z, -o, -z, -o,  o, 
  -z,  o, -h, -h,  f, -z, -z, -z,  o, -o, -o, -z, -z, -h,  o,  h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -z,  h,  h, -z, 
  -o, -z, -h,  h,  z,  f, -z,  z,  o,  o, -z,  o,  h, -z,  h, -o,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -h, -z, -z, -h, 
   h,  h, -z,  o,  z,  z,  f, -z,  o,  z,  o, -o, -o, -h, -z, -h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -h,  z, -z,  h, 
   h, -h, -o, -z,  z, -z,  z,  f,  z, -o,  o,  o, -h,  o,  h, -z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  h, -h, -z, 
  -z,  h, -o,  h,  o,  o,  o, -z,  f, -z, -z, -z, -z, -o,  h, -h, -z, -h, -h, -z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 
  -h, -z,  h,  o, -o,  o, -z, -o,  z,  f, -z,  z,  o, -z, -h, -h,  h, -z, -z,  h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 
   o, -h, -z,  h, -o,  z,  o,  o,  z,  z,  f, -z, -h,  h, -z, -o,  h,  z, -z, -h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 
  -h, -o, -h, -z,  z,  o, -o,  o,  z, -z,  z,  f,  h,  h,  o, -z,  z, -h,  h, -z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, 
   z,  z,  z,  z, -z,  h, -o, -h, -z,  o, -h,  h,  m, -z, -z, -z,  o, -o, -z,  o, -z, -h,  h, -z,  z,  z,  z,  z,  o, -z,  o, -o, 
   z,  z,  z,  z, -h, -z, -h,  o, -o, -z,  h,  h,  z,  m, -z,  z,  o,  o,  o,  z,  h, -z, -z, -h,  z,  z,  z,  z,  z,  o, -o, -o, 
   z,  z,  z,  z,  o,  h, -z,  h,  h, -h, -z,  o,  z,  z,  m, -z,  z, -o,  o, -o, -h,  z, -z, -h,  z,  z,  z,  z, -o,  o,  o, -z, 
   z,  z,  z,  z,  h, -o, -h, -z, -h, -h, -o, -z,  z, -z,  z,  m, -o, -z,  o,  o,  z,  h,  h, -z,  z,  z,  z,  z,  o,  o,  z,  o, 
   o, -z,  o,  o,  z,  z,  z,  z, -z,  h,  h, -z,  o,  o, -z, -o,  m, -z, -z, -z, -z, -o, -h, -h, -z, -h, -o,  h,  z,  z,  z,  z, 
   z,  o,  o, -o,  z,  z,  z,  z, -h, -z, -z, -h, -o,  o, -o,  z,  z,  m, -z,  z,  o, -z, -h,  h,  h, -z,  h,  o,  z,  z,  z,  z, 
  -o, -o,  o, -z,  z,  z,  z,  z, -h,  z, -z,  h,  z,  o,  o,  o,  z,  z,  m, -z,  h,  h, -z, -o,  o, -h, -z, -h,  z,  z,  z,  z, 
  -o,  o,  z,  o,  z,  z,  z,  z,  z,  h, -h, -z,  o, -z, -o,  o,  z, -z,  z,  m,  h, -h,  o, -z, -h, -o,  h, -z,  z,  z,  z,  z, 
   z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -z,  h, -h, -z, -z,  o,  h,  h,  f, -z, -z, -z,  o, -o,  o, -z, -z, -h, -o, -h, 
   z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -h, -z, -z,  h, -o, -z,  h, -h,  z,  f, -z,  z,  o,  o, -z, -o,  h, -z, -h,  o, 
   z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  h,  z, -z,  h, -h, -h, -z,  o,  z,  z,  f, -z, -o,  z,  o, -o,  o,  h, -z, -h, 
   z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -h, -h, -z, -h,  h, -o, -z,  z, -z,  z,  f,  z,  o,  o,  o,  h, -o,  h, -z, 
  -z, -h,  h, -z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -z,  h,  o, -h,  o,  o, -o, -z,  f, -z, -z, -z, -z, -o, -h,  h, 
   h, -z, -z, -h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -h, -z, -h, -o, -o,  o, -z,  o,  z,  f, -z,  z,  o, -z,  h,  h, 
  -h,  z, -z, -h,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z, -o,  h, -z,  h,  o,  z,  o,  o,  z,  z,  f, -z,  h, -h, -z, -o, 
   z,  h,  h, -z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  z,  h,  o, -h, -z,  z, -o, -o,  o,  z, -z,  z,  f, -h, -h,  o, -z, 
   o, -o, -z, -o, -z, -h, -h, -z,  z,  z,  z,  z,  o, -z, -o,  o,  z,  z,  z,  z, -z,  h,  o,  h, -z,  o,  h, -h,  m, -z, -z, -z, 
   o,  o, -o,  z,  h, -z, -z,  h,  z,  z,  z,  z,  z,  o,  o,  o,  z,  z,  z,  z, -h, -z,  h, -o, -o, -z, -h, -h,  z,  m, -z,  z, 
   z,  o,  o, -o,  h,  z, -z, -h,  z,  z,  z,  z,  o, -o,  o, -z,  z,  z,  z,  z, -o, -h, -z,  h, -h,  h, -z,  o,  z,  z,  m, -z, 
   o, -z,  o,  o,  z, -h,  h, -z,  z,  z,  z,  z, -o, -o,  z,  o,  z,  z,  z,  z, -h,  o, -h, -z,  h,  h, -o, -z,  z, -z,  z,  m;

  Eigen::SparseMatrix<flo::real> D = dense_D.sparseView();

  auto lambda = flo::host::similarity_xform(D);

  using quat_t = Eigen::Matrix<flo::real, 4, 1>;
  std::vector<quat_t> expected_lambda(8);
  expected_lambda[0] = quat_t{-0.266601,  0.266601,  0.000000,  0.051751};
  expected_lambda[1] = quat_t{ 0.159860,  0.159860,  0.000000,  0.232507};
  expected_lambda[2] = quat_t{-0.159860, -0.159860,  0.000000,  0.232507}; 
  expected_lambda[3] = quat_t{ 0.266601, -0.266601,  0.000000,  0.051751};
  expected_lambda[4] = quat_t{-0.266601, -0.266601, -0.000000,  0.051751}; 
  expected_lambda[5] = quat_t{ 0.159860, -0.159860,  0.000000,  0.232507};
  expected_lambda[6] = quat_t{-0.159860,  0.159860, -0.000000,  0.232507}; 
  expected_lambda[7] = quat_t{ 0.266601,  0.266601,  0.000000,  0.051751};

  using namespace testing;
  EXPECT_THAT(lambda, Pointwise(EigenNear(), expected_lambda));
}





