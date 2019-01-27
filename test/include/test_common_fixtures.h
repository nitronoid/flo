#ifndef FLO_INCLUDED_TEST_COMMON_FIXTURES
#define FLO_INCLUDED_TEST_COMMON_FIXTURES

#include <benchmark/benchmark.h>
#include "test_common_util.h"
#include "flo/load_mesh.hpp"

template <const char* PATH>
class SurfaceHostFixture : public benchmark::Fixture
{
public:
  void SetUp(const benchmark::State& state)
  {
    surf = flo::load_mesh(PATH);
    adjacency.resize(surf.n_faces() * 3);
    valence.resize(surf.n_vertices());
    cumulative_valence.resize(surf.n_vertices() + 1);
  }
  void TearDown(const benchmark::State& state)
  {
    surf.faces.clear();
    surf.vertices.clear();
    adjacency.clear();
    valence.clear();
    cumulative_valence.clear();
  }
  flo::Surface surf;
  // Declare arrays to dump our results
  std::vector<int> adjacency;
  std::vector<int> valence;
  std::vector<int> cumulative_valence;
};

template <const char* PATH>
class SurfaceDeviceFixture : public benchmark::Fixture
{
public:
  void SetUp(const benchmark::State& state)
  {
    surf = flo::load_mesh(PATH);
    d_face_verts.resize(surf.n_faces() * 3);
    thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());
    d_adjacency.resize(surf.n_faces() * 3);
    d_valence.resize(surf.n_vertices());
    d_cumulative_valence.resize(surf.n_vertices() + 1);
  }
  void TearDown(const benchmark::State& state)
  {
    surf.faces.clear();
    surf.vertices.clear();
    d_face_verts.clear();
    d_adjacency.clear();
    d_valence.clear();
    d_cumulative_valence.clear();
  }

  flo::Surface surf;
  thrust::device_vector<int> d_face_verts;
  // Declare device side arrays to dump our results
  thrust::device_vector<int> d_adjacency;
  thrust::device_vector<int> d_valence;
  thrust::device_vector<int> d_cumulative_valence;
};


#endif//FLO_INCLUDED_TEST_COMMON_FIXTURES
