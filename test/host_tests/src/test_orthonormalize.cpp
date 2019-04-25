//#include "test_common.h"
//#include <igl/per_vertex_normals.h>
//
//namespace
//{
//void test(std::string name)
//{
//  const std::string mp = "../matrices/" + name;
//  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
//
//  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> N;
//  igl::per_vertex_normals(
//    surf.vertices, surf.faces, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
//
//  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> expected_N(8, 3);
//  // clang-format off
//  expected_N <<
//    -0.57735, -0.57735,  0.57735,
//     0.57735, -0.57735,  0.57735,
//    -0.57735,  0.57735,  0.57735,
//     0.57735,  0.57735,  0.57735,
//    -0.57735,  0.57735, -0.57735,
//     0.57735,  0.57735, -0.57735,
//    -0.57735, -0.57735, -0.57735,
//     0.57735, -0.57735, -0.57735;
//  // clang-format on
//
//  EXPECT_MAT_NEAR(N, expected_N);
//}
//}  // namespace
//
//#define FLO_VERTEX_NORMALS_TEST(NAME) \
//  TEST(VertexNormals, NAME)           \
//  {                                   \
//    test(#NAME);                      \
//  }
//
//FLO_VERTEX_NORMALS_TEST(cube)
//// FLO_VERTEX_NORMALS_TEST(spot)
//
//#undef FLO_VERTEX_NORMALS_TEST

