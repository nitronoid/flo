#include "flo/device/surface.cuh"
#include "flo/device/cu_raii.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API std::size_t Surface::n_vertices() const noexcept
{
  return vertices.size() / 3;
}

FLO_API std::size_t Surface::n_faces() const noexcept
{
  return faces.size() / 3;
}

namespace
{
template <typename T>
void strided_copy_async(const T* i_src,
                        T* i_dest,
                        int stride,
                        int n,
                        cudaStream_t stream)
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
  cusp::array1d<real, cusp::device_memory> d_vertices(nvertices * 3);

  const int nfaces = i_host_surface.n_faces();
  // Device memory for faces to return
  cusp::array1d<int, cusp::device_memory> d_faces(nfaces * 3);

  // Scope for streams
  {
    // Get raw pointers to the vertex and face data
    auto h_vert_ptr = (flo::real*)(&i_host_surface.vertices[0][0]);
    auto h_face_ptr = (int*)(&i_host_surface.faces[0][0]);

    // Strided pointers into the device arrays
    auto d_vert_ptr_x = d_vertices.data().get() + nvertices * 0;
    auto d_vert_ptr_y = d_vertices.data().get() + nvertices * 1;
    auto d_vert_ptr_z = d_vertices.data().get() + nvertices * 2;

    auto d_face_ptr_0 = d_faces.data().get() + nfaces * 0;
    auto d_face_ptr_1 = d_faces.data().get() + nfaces * 1;
    auto d_face_ptr_2 = d_faces.data().get() + nfaces * 2;

    // Create a new stream for each copy we're going to make,
    // these are synchronized in the destructor
    ScopedCuStream s0, s1, s2, s3, s4, s5;

    // Copy the vertex information from row major to column major
    strided_copy_async(h_vert_ptr + 0, d_vert_ptr_x, 3, nvertices, s0);
    strided_copy_async(h_vert_ptr + 1, d_vert_ptr_y, 3, nvertices, s1);
    strided_copy_async(h_vert_ptr + 2, d_vert_ptr_z, 3, nvertices, s2);

    // Copy the face information from row major to column major
    strided_copy_async(h_face_ptr + 0, d_face_ptr_0, 3, nfaces, s3);
    strided_copy_async(h_face_ptr + 1, d_face_ptr_1, 3, nfaces, s4);
    strided_copy_async(h_face_ptr + 2, d_face_ptr_2, 3, nfaces, s5);
  }

  // Move all of our data into a surface struct, NOTE: this is not preventing
  // copy elision, as the surface struct itself is not being moved.
  return flo::device::Surface{std::move(d_vertices),
                              std::move(d_faces)};
}

FLO_DEVICE_NAMESPACE_END

