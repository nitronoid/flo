#include "test_common.h"

#include "flo/host/vertex_normals.hpp"

TEST(VertexNormals, cube)
{
  auto cube = make_cube();

  auto normals = flo::host::vertex_normals(cube.vertices, cube.faces);


  using normal_t = Eigen::Matrix<flo::real, 3, 1>;
  std::vector<normal_t> expected_normals {
    {-0.57735, -0.57735,  0.57735},
    { 0.57735, -0.57735,  0.57735},
    {-0.57735,  0.57735,  0.57735},
    { 0.57735,  0.57735,  0.57735},
    {-0.57735,  0.57735, -0.57735},
    { 0.57735,  0.57735, -0.57735},
    {-0.57735, -0.57735, -0.57735},
    { 0.57735, -0.57735, -0.57735}};

  using namespace testing;
  EXPECT_THAT(normals, Pointwise(EigenNear2(), expected_normals));

}



