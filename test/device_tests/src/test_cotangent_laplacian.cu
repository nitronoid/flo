#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include "flo/device/cotangent_laplacian.cuh"

TEST(CotangentLaplacian, cube)
{
  //Eigen::Matrix<flo::real, 8, 8> dense_L(8,8);
  //dense_L <<
  //   3, -1, -1,  0, -0,  0, -1, -0, 
  //  -1,  3, -0, -1,  0,  0,  0, -1, 
  //  -1, -0,  3, -1, -1,  0,  0,  0, 
  //   0, -1, -1,  3, -0, -1,  0, -0, 
  //  -0,  0, -1, -0,  3, -1, -1,  0, 
  //   0,  0,  0, -1, -1,  3, -0, -1, 
  //  -1,  0,  0,  0, -1, -0,  3, -1, 
  //  -0, -1,  0, -0,  0, -1, -1,  3;
  //Eigen::SparseMatrix<flo::real> expected_L = dense_L.sparseView();


  //------------------------------------------------------------------------
  
  // cube faces all have area (1*1)/2 = 0.5
  std::vector<flo::real> h_area(12, 0.5);
  thrust::device_vector<flo::real> d_area = h_area;

  auto cube = make_cube();
  auto raw_vert_ptr = (flo::real3*)(&cube.vertices[0][0]);
  auto raw_face_ptr = (int3*)(&cube.faces[0][0]);

  thrust::device_vector<int3> d_faces(cube.n_faces());
  thrust::copy(raw_face_ptr, raw_face_ptr + cube.n_faces(), d_faces.data());

  thrust::device_vector<flo::real3> d_verts(cube.n_vertices());
  thrust::copy(raw_vert_ptr, raw_vert_ptr + cube.n_vertices(), d_verts.data());

  uint total_valence = 36;

  auto d_L = flo::device::cotangent_laplacian(
      d_verts.data(),
      d_faces.data(),
      d_area.data(),
      cube.n_vertices(),
      cube.n_faces(),
      total_valence);

  cusp::print(d_L);

  //auto ntrip = cube.n_faces()*12;
  //thrust::device_vector<int> I(ntrip);
  //thrust::device_vector<int> J(ntrip);
  //thrust::device_vector<flo::real> V(ntrip);

  //dim3 block_dim;
  //block_dim.x = 1024 / 12; 
  //block_dim.y = 4;
  //block_dim.z = 3;
  //size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;
  //size_t nblocks = ntrip / nthreads_per_block + 1;
  //size_t shared_memory_size = sizeof(flo::real)*block_dim.x*4;

  //std::cout<<"Launch Config: \n";
  //std::cout<<"Block dim, x,y,z: ("<<block_dim.x<<", "<<block_dim.y<<", "<<block_dim.z<<")\n";
  //std::cout<<"Threads per block: "<<nthreads_per_block<<'\n';
  //std::cout<<"Num Blocks: "<<nblocks<<'\n';
  //std::cout<<"Shared memory size"<<shared_memory_size<<" (bytes)\n";

  //flo::device::d_build_triplets<<<nblocks, block_dim, shared_memory_size>>>(
  //    d_verts.data(),
  //    d_faces.data(),
  //    d_area.data(),
  //    cube.n_faces(),
  //    I.data(),
  //    J.data(),
  //    V.data());

  //auto h_I = device_vector_to_host(I);
  //auto h_J = device_vector_to_host(J);
  //auto h_V = device_vector_to_host(V);
  //for (int i = 0; i < ntrip; ++i)
  //{
  //  std::cout<<"("<<h_I[i]<<", "<<h_J[i]<<") := "<<h_V[i]<<'\n';
  //}

}



