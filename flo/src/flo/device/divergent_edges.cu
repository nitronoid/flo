#include "flo/device/divergent_edges.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__device__ real4 conjugate(const real4& i_quat)
{
  return make_float4(-i_quat.x, -i_quat.y, -i_quat.z, i_quat.w);
}

__device__ real4 hammilton_product(const real4& i_rhs, const real4& i_lhs)
{
  const real a1 = i_rhs.w;
  const real b1 = i_rhs.x;
  const real c1 = i_rhs.y;
  const real d1 = i_rhs.z;
  const real a2 = i_lhs.w;
  const real b2 = i_lhs.x;
  const real c2 = i_lhs.y;
  const real d2 = i_lhs.z;
  // W is last in a vector
  return make_float4(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
                     a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2);
}

using Tup3 = thrust::tuple<real, real, real>;
using Tup4 = thrust::tuple<real, real, real, real>;
using EdgeXform = thrust::tuple<Tup3, real4, real4>;

struct EdgeFunctor
{
  using Particle = thrust::tuple<Tup3, real4, int>;
  __device__ EdgeXform operator()(thrust::tuple<Particle, Particle> v)
  {
    real sign = 1.f;
    real4 a = v.get<0>().get<1>();
    real4 b = v.get<1>().get<1>();

    // We swap the ordering based on the larger index
    if (v.get<0>().get<2>() > v.get<1>().get<2>())
    {
      auto t = a;
      a = b;
      b = t;
    }
    else if (v.get<0>().get<2>() == v.get<1>().get<2>())
    {
      sign = 0.f;
    }

    return thrust::make_tuple(
      thrust::make_tuple(
        sign * (v.get<0>().get<0>().get<0>() - v.get<1>().get<0>().get<0>()),
        sign * (v.get<0>().get<0>().get<1>() - v.get<1>().get<0>().get<1>()),
        sign * (v.get<0>().get<0>().get<2>() - v.get<1>().get<0>().get<2>())),
      a,
      b);
  }
};

struct XformedEdgeFunctor
{
  __device__ Tup4 operator()(thrust::tuple<EdgeXform, real> input) const
  {
    real4 ex;
    {
      // Unpack all our parameters from the input tuple
      const real4 e = make_float4(input.get<0>().get<0>().get<0>(),
                                  input.get<0>().get<0>().get<1>(),
                                  input.get<0>().get<0>().get<2>(),
                                  0.0f);
      const real4 l1 = input.get<0>().get<1>();
      const real4 l2 = input.get<0>().get<2>();

      const real r3 = 1.f / 3.f;
      const real r6 = 1.f / 6.f;

      ex = hammilton_product(hammilton_product(r3 * conjugate(l1), e), l1) +
           hammilton_product(hammilton_product(r6 * conjugate(l1), e), l2) +
           hammilton_product(hammilton_product(r6 * conjugate(l2), e), l1) +
           hammilton_product(hammilton_product(r3 * conjugate(l2), e), l2);
    }

    const real cot_alpha = input.get<1>();
    ex *= cot_alpha * -1.f;

    return thrust::make_tuple(ex.x, ex.y, ex.z, ex.w);
  }
};

struct Tup4Add
{
  __device__ Tup4 operator()(const Tup4& lhs, const Tup4& rhs) const
  {
    return thrust::make_tuple(lhs.get<0>() + rhs.get<0>(),
                              lhs.get<1>() + rhs.get<1>(),
                              lhs.get<2>() + rhs.get<2>(),
                              lhs.get<3>() + rhs.get<3>());
  }
};
}  // namespace

FLO_API void divergent_edges(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_xform,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_cotangent_laplacian,
  cusp::array2d<real, cusp::device_memory>::view do_edges)
{
  // Iterate over the output edge x,y,z,w
  auto out_it =
    thrust::make_zip_iterator(thrust::make_tuple(do_edges.row(0).begin(),
                                                 do_edges.row(1).begin(),
                                                 do_edges.row(2).begin(),
                                                 do_edges.row(3).begin()));
  // Iterate over the transforms as packed quaternions
  auto xform_it = thrust::device_pointer_cast(
    reinterpret_cast<const real4*>(di_xform.begin().base().get()));

  // Zip iterate over x,y,z
  auto vert_it =
    thrust::make_zip_iterator(thrust::make_tuple(di_vertices.row(0).begin(),
                                                 di_vertices.row(1).begin(),
                                                 di_vertices.row(2).begin()));
  // Iterate over a vertex, it's corresponding transform and the index
  auto count_it = thrust::make_counting_iterator(0);
  auto vert_xform_it =
    thrust::make_zip_iterator(thrust::make_tuple(vert_it, xform_it, count_it));
  // Iterate over pairs of verts and their transforms composing a directed edge
  auto edge_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_permutation_iterator(
        vert_xform_it, di_cotangent_laplacian.row_indices.begin()),
      thrust::make_permutation_iterator(
        vert_xform_it, di_cotangent_laplacian.column_indices.begin()))),
    EdgeFunctor{});

  // Iterate over the vertex transform pairs, with the corresponding cotangent
  // laplacian matrix values, transforming them into divergent edges on read
  auto edge_value_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(edge_it, di_cotangent_laplacian.values.begin())),
    XformedEdgeFunctor{});

  // Reduce by adjacency
  thrust::reduce_by_key(di_cotangent_laplacian.row_indices.begin(),
                        di_cotangent_laplacian.row_indices.end(),
                        edge_value_it,
                        thrust::make_discard_iterator(),
                        out_it,
                        thrust::equal_to<int>(),
                        Tup4Add{});
}

FLO_DEVICE_NAMESPACE_END
