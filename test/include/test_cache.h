#ifndef FLO_INCLUDED_TEST_CACHE
#define FLO_INCLUDED_TEST_CACHE

#include <unordered_map>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>
#include "flo/host/surface.hpp"
#ifdef __CUDACC__
#include "flo/device/surface.cuh"
#endif
#include "flo/host/load_mesh.hpp"
#include "test_common_macros.h"

struct TestCache
{
  // Only include device surface in the cache if we target cuda
  using SurfTuple = std::tuple<std::unique_ptr<flo::host::Surface>
#ifdef __CUDACC__
                               ,std::unique_ptr<flo::device::Surface>
#endif
                               >;

  // The cache map type
  using SurfMap = std::unordered_map<std::string, SurfTuple>;
  static SurfMap m_cache;
  static const std::string m_mesh_path;

private:
  static std::mutex m_mutex;

public:
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
    decltype(*std::get<k_MEM>(SurfTuple{}))&
  {
#ifndef __CUDACC__
    static_assert(X != DEVICE, "Device surface not available without CUDA");
#endif
    std::unique_lock<std::mutex> lock(m_mutex);
    const std::string base_path = STRINGIFY(MESH_PATH);
    const std::string file_path = base_path + '/' + i_file_path;

    auto mesh_it = m_cache.find(file_path);
    if (mesh_it == m_cache.end())
    {
      std::cout << "Loading mesh from file: " << file_path << '\n';
      std::unique_ptr<flo::host::Surface> h_mesh(new flo::host::Surface);
      *h_mesh = flo::host::load_mesh(file_path.c_str());
#ifdef __CUDACC__
      std::cout << "Performing device copy\n";
      std::unique_ptr<flo::device::Surface> d_mesh(new flo::device::Surface);
      *d_mesh = flo::device::make_surface(*h_mesh);
#endif
      m_cache[file_path] = SurfTuple{std::move(h_mesh)
#ifdef __CUDACC__
                              ,std::move(d_mesh)
#endif
      };
      return *std::get<k_MEM>(m_cache[file_path]);
    }
    return *std::get<k_MEM>(mesh_it->second);
  }
};

#endif  // FLO_INCLUDED_TEST_CACHE
