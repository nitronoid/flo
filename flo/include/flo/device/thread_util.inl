__device__ __forceinline__ uint unique_thread_idx1()
{
  return
    // Global block index
    (blockIdx.x) *
      // Number of threads in a block
      (blockDim.x) +
    // thread index in block
    (threadIdx.x);
}

__device__ __forceinline__ uint unique_thread_idx2()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x) *
      // Number of threads in a block
      (blockDim.x * blockDim.y) +
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x);
}

__device__ __forceinline__ uint unique_thread_idx3()
{
  return
    // Global block index
    (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
      // Number of threads in a block
      (blockDim.x * blockDim.y * blockDim.z) +
    // thread index in block
    (threadIdx.x + threadIdx.y * blockDim.x +
     threadIdx.z * blockDim.x * blockDim.y);
}

__device__ __forceinline__ uint block_index()
{
  return blockIdx.x + blockIdx.y * gridDim.x +
         blockIdx.z * gridDim.x * gridDim.y;
}

__device__ __forceinline__ uint block_volume()
{
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ uint8_t cycle(uint8_t i_x)
{
  /************************
     mapping is as follows
     0 -> 2
     1 -> 0
     2 -> 1
  ************************/
  uint8_t c = i_x + 0xFC;
  return __ffs(c) - 1;
}

__device__ __forceinline__ uchar3 tri_edge_loop(uint8_t i_e)
{
  /************************
    e.g. input == 1
     x -> 1
     y -> 2
     z -> 0
  ************************/
  uchar3 loop;
  loop.x = i_e;
  loop.z = cycle(loop.x);
  loop.y = cycle(loop.z);
  return loop;
}

__device__ __forceinline__ uchar4 quat_loop(uint8_t i_e)
{
  /************************
    produces the hamilton product layout, adjusted for vec4 layout (w last):
    0 -> [3 0 1 2]
    1 -> [0 3 2 1]
    2 -> [1 2 3 0]
    3 -> [2 1 0 3]
  ************************/
  return make_uchar4((3u + i_e) & 3u,
          (0u + i_e + 2u * (i_e & 1u)) & 3u,
          (1u + i_e) & 3u,
          (2u + i_e + 2u * (i_e & 1u)) & 3u);
}

__device__ __forceinline__ int sign_from_bit(uint8_t i_byte, uint8_t i_bit)
{
  /************************
    extracts the requested bit from the given byte and uses it to create a 
    signed integer e.g.
    byte = 00110110,  bit = 4
    shift -> 00110110 >> 3 == 00000011
    bitwise AND -> 00000011 & 1 == 00000001
    shift -> 00000001 << 1 == 00000010
    complement -> ~00000010 == 11111101
    add -> 11111101 + 1 == 11111110
    result -> -1
  ************************/
  return (~((i_byte >> i_bit) & 1u)<<1) + 3;;
}

