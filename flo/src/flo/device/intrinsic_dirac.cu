#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/thread_util.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace
{
__device__ real4 hammilton_product(const real3& i_rhs, const real3& i_lhs)
{
  const auto a1 = 0.f;
  const auto b1 = i_rhs.x;
  const auto c1 = i_rhs.y;
  const auto d1 = i_rhs.z;
  const auto a2 = 0.f;
  const auto b2 = i_lhs.x;
  const auto c2 = i_lhs.y;
  const auto d2 = i_lhs.z;
  // W is last in a vector
  return make_float4(
      a1*b2 + b1*a2 + c1*d2 - d1*c2,
      a1*c2 - b1*d2 + c1*a2 + d1*b2,
      a1*d2 + b1*c2 - c1*b2 + d1*a2,
      a1*a2 - b1*b2 - c1*c2 - d1*d2);
}
//__device__ double atomicAdd(double* __restrict__ i_address, const double i_val)
//{
//  auto address_as_ull = reinterpret_cast<unsigned long long int*>(i_address);
//  unsigned long long int old = *address_as_ull, ass;
//  do
//  {
//    ass = old;
//    old = atomicCAS(address_as_ull,
//                    ass,
//                    __double_as_longlong(i_val + __longlong_as_double(ass)));
//  } while (ass != old);
//  return __longlong_as_double(old);
//}

template <typename T>
__device__ constexpr T reciprocal(T&& i_value) noexcept
{
  return T{1} / i_value;
}
// block dim should be 3*#F, where #F is some number of faces,
// we have three edges per triangle face, and write two values per edge
__global__ void
d_intrinsic_dirac_atomic(const real3* __restrict__ di_vertices,
                         const int* __restrict__ di_faces,
                         const real* __restrict__ di_face_area,
                         const real* __restrict__ di_rho,
                         const int* __restrict__ di_cumulative_valence,
                         const int* __restrict__ di_entry_offset,
                         const uint i_nfaces,
                         int* __restrict__ do_rows,
                         int* __restrict__ do_columns,
                         real4* __restrict__ do_values)
{
  // Declare one shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Create pointers into the block dividing it for the different uses
  real* __restrict__ face_area = (real*)shared_memory;
  // There is a cached value for each corner of the face so we offset
  real3* __restrict__ edges = (real3*)(face_area + blockDim.x * 3);
  // There are nfaces *3 vertex values (duplicated for each face vertex)
  real* __restrict__ edge_norm2 = (real*)(edges + blockDim.x * 3);
  // There are nfaces *3 squared edge lengths (duplicated for each face vertex)
  real* __restrict__ rho = (real*)(edge_norm2 + blockDim.x * 3);
  // There are nfaces *3 vertex rho values (duplicated for each face vertex)
  uint32_t* __restrict__ eid = (uint32_t*)(rho + blockDim.x * 3);

  // Calculate which face this thread is acting on
  const uint fid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check we're not out of range
  if (fid >= i_nfaces)
    return;

  // Get the vertex order, need to half the tid as we have two threads per edge
  const uchar3 loop = edge_loop(threadIdx.y >> 1);

  // Compute local edge indices rotated by the corner this thread corresponds to
  const uint16_t local_e0 = threadIdx.x * 3 + loop.x;
  const uint16_t local_e1 = threadIdx.x * 3 + loop.y;
  const uint16_t local_e2 = threadIdx.x * 3 + loop.z;

  // This thread will write to column or row major triangle based on even or odd
  const uint8_t major = !(threadIdx.y & 1);

  // Only write once per face
  if (!threadIdx.y)
  {
    // Duplicate for each corner of the face to reduce bank conflicts
    face_area[local_e0] = face_area[local_e1] = face_area[local_e2] =
      di_face_area[fid];
  }
  // Write the vertex positions into shared memory
  if (major)
  {
    const uint32_t vid = di_faces[fid * 3 + loop.x];
    rho[local_e0] = di_rho[vid];
    edges[local_e0] = di_vertices[vid];
  }
  __syncthreads();
  // Compute squared length of edges and write to shared memory
  if (major)
  {
    edges[local_e0] = edges[local_e2] - edges[local_e1];
  }
  __syncthreads();
  const uint16_t O1 = threadIdx.x * 3 + nth_element(loop, 1 + major);
  const uint16_t O2 = threadIdx.x * 3 + nth_element(loop, 1 + !major);
  // Calc imaginary part
  real4 value =
    hammilton_product(edges[O1], edges[O2]) / (-4.f * face_area[local_e0]);
  value += make_float4(
    reciprocal(6.f) * (rho[O1] * edges[O2] - rho[O2] * edges[O1]));
  // Add real part
  value.w +=
    rho[O1] * rho[O2] * face_area[local_e0] * reciprocal(9.f);

  // Write the opposing edge ID's into shared memory to reduce global reads
  eid[local_e0 * 2 + !major] =
    di_faces[fid * 3 + nth_element(loop, 1 + !major)];
  __syncthreads();

  const uint32_t R = eid[local_e0 * 2 + !major];
  const uint32_t C = eid[local_e0 * 2 + major];
  const uint32_t address = di_entry_offset[fid * 6 + threadIdx.y] + R + (C > R);
  // Write the row and column indices
  do_rows[address] = R;
  do_columns[address] = C;
  auto out = reinterpret_cast<real*>(do_values + address);
  atomicAdd(out + 0, value.x);
  atomicAdd(out + 1, value.y);
  atomicAdd(out + 2, value.z);
  atomicAdd(out + 3, value.w);
}

}  // namespace

void intrinsic_dirac(const thrust::device_ptr<const real3> di_vertices,
                     const thrust::device_ptr<const int3> di_faces,
                     const thrust::device_ptr<const real> di_face_area,
                     const thrust::device_ptr<const real> di_rho,
                     const thrust::device_ptr<const int> di_cumulative_valence,
                     const thrust::device_ptr<const int2> di_entry_offset,
                     const int i_nverts,
                     const int i_nfaces,
                     const int i_total_valence,
                     thrust::device_ptr<int> do_diagonals,
                     thrust::device_ptr<int> do_rows,
                     thrust::device_ptr<int> do_columns,
                     thrust::device_ptr<real4> do_values)
{
  dim3 block_dim;
  block_dim.y = 6;
  block_dim.x = 170;
  size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t nblocks = i_nfaces * 6 / nthreads_per_block + 1;
  // face area | cot_alpha  =>  sizeof(real) * 3 * #F
  // vertex positions       =>  sizeof(real3) * 3 * #F ==  sizeof(real) * 9 * #F
  // edge squared lengths   =>  sizeof(real) * 3 * #F
  // === (3 + 9 + 3) * #F * sizeof(real)
  size_t shared_memory_size =
    sizeof(flo::real) * block_dim.x * 18 + sizeof(uint32_t) * 6 * block_dim.x;

  // When passing the face and offset data to cuda, we reinterpret them as int
  // arrays. The advantage of this is coalesced memory reads by neighboring
  // threads, and access at a more granular level.
  // The cast is inherently safe due to the alignment of cuda vector types,
  // and reinterpret casting guarantees no changes to the underlying values
  d_intrinsic_dirac_atomic<<<nblocks, block_dim, shared_memory_size>>>(
    di_vertices.get(),
    reinterpret_cast<const int*>(di_faces.get()),
    di_face_area.get(),
    di_rho.get(),
    di_cumulative_valence.get(),
    reinterpret_cast<const int*>(di_entry_offset.get()),
    i_nfaces,
    do_rows.get(),
    do_columns.get(),
    do_values.get());
  cudaDeviceSynchronize();

  thrust::counting_iterator<int> counter(0);
  thrust::copy_if(
    counter + di_cumulative_valence[1] + 1,
    counter + i_total_valence + i_nverts,
    do_diagonals + 1,
    [do_rows = do_rows.get()] __device__(int x) { return !do_rows[x]; });

  // Iterator for diagonal matrix entries
  auto diag_begin = thrust::make_permutation_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(do_rows, do_columns)),
    do_diagonals);

  // Generate the diagonal entry, row and column indices
  thrust::transform(
    counter, counter + i_nverts, diag_begin, [] __device__(const int i) {
      return thrust::make_tuple(i, i);
    });

}

FLO_DEVICE_NAMESPACE_END

