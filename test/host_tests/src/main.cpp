#include "gtest/gtest.h"
#include "test_cache.h"

TestCache::SurfMap TestCache::m_cache;
std::mutex TestCache::m_mutex;

int main(int argc, char** argv)
{
  TestCache::get_mesh<TestCache::HOST>("cube.obj");
  TestCache::get_mesh<TestCache::HOST>("spot.obj");
  TestCache::get_mesh<TestCache::HOST>("dense_sphere_400x400.obj");
  TestCache::get_mesh<TestCache::HOST>("dense_sphere_1000x1000.obj");
  // TestCache::get_mesh<TestCache::HOST>("dense_sphere_1500x1500.obj");
  // TestCache::get_mesh<TestCache::HOST>("cube_1k.obj");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
