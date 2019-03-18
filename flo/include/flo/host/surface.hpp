#ifndef FLO_HOST_INCLUDED_SURFACE
#define FLO_HOST_INCLUDED_SURFACE

#include "flo/flo_internal.hpp"
#include <Eigen/Dense>
#include <vector>

FLO_HOST_NAMESPACE_BEGIN

struct Surface
{
  std::vector<Eigen::Matrix<real, 3, 1>> vertices;
  std::vector<Eigen::Vector3i> faces;

  FLO_API std::size_t n_vertices() const noexcept;
  FLO_API std::size_t n_faces() const noexcept;
};

template <bool CONST>
struct SurfaceView
{
private:
  template <bool ISCONST, typename T>
  using conditional_const_t =
    typename std::conditional<ISCONST,
                              typename std::add_const<T>::type,
                              typename std::remove_const<T>::type>::type;

public:
  SurfaceView(const SurfaceView&) = default;
  SurfaceView& operator=(const SurfaceView&) = default;
  SurfaceView(SurfaceView&&) = default;
  SurfaceView& operator=(SurfaceView&&) = default;
  ~SurfaceView() = default;

  template <typename DUMMY = void,
            typename = typename std::enable_if<!CONST, DUMMY>::type>
  SurfaceView(gsl::span<Eigen::Matrix<real, 3, 1>> i_vertices,
              gsl::span<Eigen::Vector3i> i_faces)
    : vertices(std::move(i_vertices)), faces(std::move(i_faces))
  {
  }

  // Create a const surface view from non-const or const spans
  template <typename REAL3,
            typename INT3,
            typename DUMMY = void,
            typename = typename std::enable_if<
              CONST &&
                std::is_same<typename std::remove_const<REAL3>::type,
                             Eigen::Matrix<real, 3, 1>>::value &&
                std::is_same<typename std::remove_const<INT3>::type,
                             Eigen::Vector3i>::value,
              DUMMY>::type>
  SurfaceView(gsl::span<REAL3> i_vertices, gsl::span<INT3> i_faces)
    : vertices(std::move(i_vertices)), faces(std::move(i_faces))
  {
  }

  // Create a const surface view from a non-const
  template <typename DUMMY = void,
            typename = typename std::enable_if<CONST, DUMMY>::type>
  SurfaceView(const SurfaceView<false>& i_sv)
    : vertices(i_sv.vertices), faces(i_sv.faces)
  {
  }

  // Create a const surface view from a non-const
  template <typename DUMMY = void,
            typename = typename std::enable_if<CONST, DUMMY>::type>
  SurfaceView& operator=(const SurfaceView<false>& i_sv)
  {
    vertices = i_sv.vertices;
    faces = i_sv.faces;
  }

  FLO_API std::size_t n_vertices() const noexcept
  {
    return vertices.size();
  }

  FLO_API std::size_t n_faces() const noexcept
  {
    return faces.size();
  }

  conditional_const_t<
    CONST,
    gsl::span<conditional_const_t<CONST, Eigen::Matrix<real, 3, 1>>>>
    vertices;
  conditional_const_t<CONST,
                      gsl::span<conditional_const_t<CONST, Eigen::Vector3i>>>
    faces;
};

using SurfaceViewMutable = SurfaceView<false>;
using SurfaceViewImmutable = SurfaceView<true>;

FLO_API Surface make_surface(const ::flo::host::Surface& i_host_surface);

FLO_API SurfaceViewMutable make_surface_view(::flo::host::Surface& i_surface);

FLO_API SurfaceViewImmutable
make_surface_view(const ::flo::host::Surface& i_surface);

FLO_HOST_NAMESPACE_END

#endif  // FLO_HOST_INCLUDED_SURFACE
