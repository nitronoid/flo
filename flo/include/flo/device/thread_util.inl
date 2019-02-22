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

__device__ __forceinline__ uchar3 edge_loop(uint8_t i_e)
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

