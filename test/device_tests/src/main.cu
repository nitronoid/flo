#include "gtest/gtest.h"
#include "test_cache.h"

TestCache::SurfMap TestCache::m_cache;

int main(int argc, char** argv)
{
  std::cout<<"Begin tests\n";
  TestCache::get_mesh<TestCache::HOST>("cube.obj");
  TestCache::get_mesh<TestCache::HOST>("spot.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_400x400.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_1000x1000.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_1500x1500.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("cube_1k.obj");
  std::cout<<"Caching complete\n";

  ::testing::InitGoogleTest(&argc, argv);
  auto RESULT = RUN_ALL_TESTS();

  // Necessary to manually free all device memory,
  // as static variables will be free'd after cuda runtime has already shut down
  TestCache::m_cache.clear();
  return RESULT;
}
