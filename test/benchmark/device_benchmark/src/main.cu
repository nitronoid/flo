#include <benchmark/benchmark.h>
#include "test_cache.h"

TestCache::SurfMap TestCache::m_cache;
std::mutex TestCache::m_mutex;

int main(int argc, char** argv) 
{
  benchmark::Initialize(&argc, argv);
  std::cout<<"Loading and caching test meshes.\n";
  TestCache::get_mesh<TestCache::DEVICE>("cube.obj");
  TestCache::get_mesh<TestCache::DEVICE>("spot.obj");
  TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_400x400.obj");
  TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_1000x1000.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("dense_sphere_1500x1500.obj");
  //TestCache::get_mesh<TestCache::DEVICE>("cube_1k.obj");
  std::cout<<"Test mesh caching complete\n";

  benchmark::RunSpecifiedBenchmarks();

  TestCache::m_cache.clear();
}
