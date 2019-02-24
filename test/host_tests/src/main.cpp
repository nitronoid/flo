#include "gtest/gtest.h"
#include "test_cache.h"

TestCache::SurfMap TestCache::m_cache;

int main(int argc, char** argv)
{
  TestCache::get_mesh<TestCache::HOST>("../models/cube.obj");
  TestCache::get_mesh<TestCache::HOST>("../models/spot.obj");
  TestCache::get_mesh<TestCache::HOST>("../models/dense_sphere_400x400.obj");
  TestCache::get_mesh<TestCache::HOST>("../models/dense_sphere_1000x1000.obj");
  // TestCache::get_mesh<TestCache::HOST>("../models/dense_sphere_1500x1500.obj");
  // TestCache::get_mesh<TestCache::HOST>("../models/cube_1k.obj");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
