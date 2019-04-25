#include "flo/device/surface.cuh"
#include "flo/device/cu_raii.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API std::size_t Surface::n_vertices() const noexcept
{
  return vertices.num_cols;
}

FLO_API std::size_t Surface::n_faces() const noexcept
{
  return faces.num_cols;
}

namespace
{
template <typename T>
void strided_copy_async(
  const T* i_src, T* i_dest, int stride, int n, cudaStream_t stream)
{
  cudaMemcpy2DAsync(i_dest,
                    sizeof(T),
                    i_src,
                    sizeof(T) * stride,
                    sizeof(T),
                    n,
                    cudaMemcpyHostToDevice,
                    stream);
}
}  // namespace

FLO_API Surface make_surface(const ::flo::host::Surface& i_host_surface)
{
  const int nvertices = i_host_surface.n_vertices();
  // Device memory for vertices to return
  cusp::array2d<real, cusp::device_memory> d_vertices(3, nvertices);

  const int nfaces = i_host_surface.n_faces();
  // Device memory for faces to return
  cusp::array2d<int, cusp::device_memory> d_faces(3, nfaces);

  auto h_vert_ptr = i_host_surface.vertices.data();
  thrust::copy(
    h_vert_ptr, h_vert_ptr + nvertices * 3, d_vertices.values.begin());
  auto h_face_ptr = i_host_surface.faces.data();
  thrust::copy(h_face_ptr, h_face_ptr + nfaces * 3, d_faces.values.begin());

  // Move all of our data into a surface struct, NOTE: this is not preventing
  // copy elision, as the surface struct itself is not being moved.
  return flo::device::Surface{std::move(d_vertices), std::move(d_faces)};
}

FLO_DEVICE_NAMESPACE_END

