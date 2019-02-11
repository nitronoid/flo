#include <benchmark/benchmark.h>
#include "test_cache.h"

std::unordered_map<std::string, flo::Surface> TestCache::m_cache;

int main(int argc, char** argv) 
{
  benchmark::Initialize(&argc, argv);
  std::cout<<"Loading and caching test meshes.\n";
  TestCache::get_mesh("../models/cube.obj");
  TestCache::get_mesh("../models/spot.obj");
  TestCache::get_mesh("../models/dense_sphere_400x400.obj");
  TestCache::get_mesh("../models/dense_sphere_1000x1000.obj");
  //TestCache::get_mesh("../models/dense_sphere_1500x1500.obj");
  //TestCache::get_mesh("../models/cube_1k.obj");
  std::cout<<"Test mesh caching complete\n";
  benchmark::RunSpecifiedBenchmarks();
}
