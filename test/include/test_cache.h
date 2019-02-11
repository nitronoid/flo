#ifndef FLO_INCLUDED_TEST_CACHE
#define FLO_INCLUDED_TEST_CACHE

#include <unordered_map>
#include <string>
#include <iostream>
#include "flo/surface.hpp"
#include "flo/load_mesh.hpp"

class TestCache
{
  static std::unordered_map<std::string, flo::Surface> m_cache;

public:
  inline static flo::Surface& get_mesh(const std::string& i_file_path)
  {
    auto mesh_it = m_cache.find(i_file_path);
    if (mesh_it == m_cache.end())
    {
      std::cout<<"Loading mesh from file: "<<i_file_path<<'\n';
      m_cache[i_file_path] = flo::load_mesh(i_file_path.c_str());
      return m_cache[i_file_path];
    }
    return mesh_it->second;
  }
};


#endif//FLO_INCLUDED_TEST_CACHE
