#include "gtest/gtest.h"
#include "test_cache.h"

TestCache::SurfMap TestCache::m_cache;
std::mutex TestCache::m_mutex;

int main(int argc, char** argv)
{
  TestCache::get_mesh<TestCache::HOST>("cube.obj");
  TestCache::get_mesh<TestCache::HOST>("spot.obj");
  TestCache::get_mesh<TestCache::HOST>("bunny.obj");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
