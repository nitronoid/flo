#ifndef FLO_INCLUDED_TEST_CACHE
#define FLO_INCLUDED_TEST_CACHE

#include <unordered_map>
#include <string>
#include <iostream>
#include "flo/host/surface.hpp"
#ifdef __CUDACC__
#include "flo/device/surface.cuh"
#endif
#include "flo/host/load_mesh.hpp"

struct TestCache
{
  // Only include device surface in the cache if we target cuda
  using SurfTuple = std::tuple<flo::host::Surface
#ifdef __CUDACC__
                               ,flo::device::Surface
#endif
                               >;

  // The cache map type
  using SurfMap = std::unordered_map<std::string, SurfTuple>;
  static SurfMap m_cache;

  // For the user to indicate which memory space they wish to access
  enum SURFACEMEM { HOST = 0, DEVICE = 1 };

  template <SURFACEMEM X,
#ifdef __CUDACC__
            int k_MEM = X
#else
            int k_MEM = 0
#endif
            >
  inline static auto get_mesh(const std::string& i_file_path) ->
    typename std::tuple_element<k_MEM, SurfTuple>::type&
  {
#ifndef __CUDACC__
    static_assert(X != DEVICE, "Device surface not available without CUDA");
#endif

    auto mesh_it = m_cache.find(i_file_path);
    if (mesh_it == m_cache.end())
    {
      std::cout << "Loading mesh from file: " << i_file_path << '\n';
      auto h_mesh = flo::host::load_mesh(i_file_path.c_str());
#ifdef __CUDACC__
      std::cout << "Performing device copy\n";
      auto d_mesh = flo::device::make_surface(h_mesh);
#endif
      m_cache[i_file_path] = {std::move(h_mesh)
#ifdef __CUDACC__
                              ,std::move(d_mesh)
#endif
      };
      return std::get<k_MEM>(m_cache[i_file_path]);
    }
    return std::get<k_MEM>(mesh_it->second);
  }
};

#endif  // FLO_INCLUDED_TEST_CACHE
