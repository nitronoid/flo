# Conformal Willmore flow
Based on the work of Keenan Crane, this is an implementation of Conformal curvature flow using spin transformations.

## Dependencies
- Eigen (header only, version 3.3.7)
- IGL (header only)
- qmake (version 5)

#### Dependencies (Device code only)
- CUDA (version 9.0 or greater)
- Thrust (shipped with CUDA)
- CUSP (header only)

#### Dependencies (Tests and benchmarks)
- Google Test
- Google Benchmark


## Build
### Configuration
The following environment variables are required for compilation:

CUDA_PATH : Needs to point to the base directory of your cuda lib and includes
CUDA_ARCH : Your device compute capability (e.g. 30, 35, ... 50, 52 ...)
HOST_COMPILER : Your local g++ compiler (compatible with cuda compiles, g++ 4.8.5)


```
> git submodule update --init --recursive
> qmake
> make -j
```

## References:
    [1] K. Crane, U. Pinkall, and P. Schröder, “Spin transformations of discrete surfaces,” ACM Transactions on Graphics, vol. 30, no. 4, p. 1, Jul. 2011.
        
    [2] K. Crane, U. Pinkall, and P. Schröder, “Robust fairing via conformal curvature flow,” ACM Transactions on Graphics, vol. 32, no. 4, p. 1, Jul. 2013.
                

## Project Structure

- The library contains a host and a device implementation, separated by namspaces.
- The host implementation is header only and was designed to feel like an extension of the popular IGL library (it follows their style guide too.)
- The device implementation requires compiling into a shared library and has been R-pathed into the tests and benchmarks for convenience.
- The project comes with 2 host samples, and one device sample. The first host, and only device sample is a demo implementation of willmore flow. The second host sample named matrix_bake, will export all matrices required by the tests and benchmarks in matrix market format.
- The project comes with a full set of host and device tests and benchmarks.
- Currently all subdirectories dump an executable in their directory.
- NOTE: Spin positions test is failing for the DEVICE side implementation as it finds an equivalent but different solution to the linear system

## Development of parallel implementation
This project was developed as a final year programming assignment, with the goal of reimplementing a sequential algorithm on the GPU, and benchmarking the performance gains.
I chose to implement conformal willmore flow using spin tranformations as nature of a geometry processing project lends itself to data oriented design. I have outlined some of the more noteworthy parts of the project below.

<details>
<summary><b>Adjacency Matrix Indices</b></summary>
<p>

### Overview

One of the main aspects of the project was to assemble sparse matrices in parallel. One possible approach would be to compute lists of triplets, which could then be sorted by row and column index,
before being reduced. The draw back of this approach is of course using two sorts. The two sparse matrices required for this project both share the same sparsity pattern (laplacian), and we know that,
all entries in the matrix will only be written to a maximum of twice (except the diagonals), so it makes sense to try and use atomics here as the number of collisions will be very low.
To facilitate this, each thread working on the matrix would require an index to write out it's value, and this function provides that.

The goal is to produce the matrix column and row indices, as well as diagonal indices (useful for later), and entry indices for parallel assembly. The first two are essentially the vertex-vertex adjacency
lists with added diagonals. The adjacency lists can be found by creating a list of all possible half edges and then using unique to remove duplicates. Once we've obtained them, we can find the locations where our diagonal entries should be inserted by comparing each row index against each column index and writing a one if, it is less. If we follow this with a reduce, we've obtained the amount of rows
per column before the diagonal, so to finish we can add an ascending list and the cumulative valence of each row (prior to diagonals being inserted) to these reduced values and voi-la.

```
row_index: 0 0 0 1 1 2 2 3 3 3
col_index: 1 2 3 0 3 0 3 0 1 2
--------------------------------
compared : 0 0 0 1 0 1 0 1 1 1
--------------------------------
index    : 0 1 2 3
reduced  : 0 1 1 3
--------------------------------
valence  : 3 2 2 3
cumval   : 0 3 5 7 10
sequence : 0 1 2 3
(reduced + cumval + sequence =)
diagonals: 0 5 8 13
```
Handy optimizations are to pack the comparisson and reduce into one kernel using a thrust transform iterator on the way in, and then pack the final sequence and valence add in to the end using a transform output iterator, leaving us with one single kernel launch.

Given the diagonals, it is simple to produce the row and column indices by copying across the vertex-vertex adjacency lists, but making sure to skip our diagonals. Then in a second pass, the diagonals indices
are copied to using an ascending sequence. In practice this can all be done with thrust and permutation iterators. 
```
row_index: 0 0 0 1 1 2 2 3 3 3
col_index: 1 2 3 0 3 0 3 0 1 2
diagonals: 0 5 8 13
--------------------------------------
1st pass : - 0 0 0 1 - 1 2 - 2 3 3 3 -
2nd pass : 0 0 0 0 1 1 1 2 2 2 3 3 3 3
```


As for the regular entry indices, we use a parallel binary search method on each in each face (same order as the threads in the sparse matrix assembly kernels.)
Given an edge A -> B we find the upper bound and lower bounds as the cumulative valence of vertex A, and one past A. We then perform a binary search within these bounds for B.
Unfortunately this can yeild unbalanced work-loads for threads in the kernel, however it need only be computed once and executes quick enough for this purpose, especially when combined with the gains
in the sparse matrix assembly.

```
row_index   : 0 0 0 1 1 2 2 3 3 3
col_index   : 1 2 3 0 3 0 3 0 1 2
cumval      : 0 3 5 7 10
------------------------------
f0          : 0 1 3
lower_bound : 0 3 7
upper_bound : 3 5 10
b-search    : 0 4 7
```

</p>
</details>

<details>
<summary><b>Cotangent Laplacian</b></summary>
<p>

### Overview

The cotangent laplacian is a very useful discrete representation of a surfaces mean curvature, used heavily for this algorithm. It will be recomputed at every iteration, and so it was essential
that the assembly be quick. Initially I began by launching 6 threads per face, meaning 2 threads per face edge, which directly corresponds to the amount of writes into the matrix required.
This allowed for one atomic add per thread, and hence a fairly quick assembly, however the uneven number of threads per block, and the fact that threads working on the same face could be in different
warps limited my ability to combat bank conflicts. My original solution was to instead have 8 threads per face, with 2 dummy threads doing no work, but aligning the rest. However I then realised
that once the alignment was in place, the entire computation could be performed using only interwarp communcation rather than block-wide. This would mean I could get rid of shared memory and potential back conflicts, and instead use warp shuffling to share data between my threads. At this point it seemed wasteful to have 2 dead threads, so I settled for 4 threads per face, with one dead thread. Each alive thread would write to global memory twice, once in the upper triangular part and once in the lower.
The thread communication is fairly simple, each thread loads in a vertex positon (1-per thread), and then using a warp shuffle, we subtract the neighbours vertex value to produce 3 edges.
With these edges I first compute the face area, and then the final cotangent laplacian value. Writing these value back to global memory is simple since we have computed the entry indices in the previous step.

```
if lane < 3:
    read vertex_pos

shuffle (0 1 2 -) -> (1 2 0 1)
shuffle down subtract (1-2 2-0 0-1 1-2)
shuffle donw neighbor_edge
inv_area <- 0.5 * rsqrt(cross(edge, neighbor_edge))
result <- dot(edge, neigbor_edge) * inv_area

```
In the real implementation I don't just kill off the 4th thread, instead I use it as temporary storage to make shuffle downs easier.
The main kernel leaves the diagonals untouched, as they are easier computed in a second pass, we need simply reduce each column and write the negation of the sum.


</p>
</details>

<details>
<summary><b>Intrinsic Dirac</b></summary>
<p>

### Overview

The intrinsic dirac operator is the core of the algorithm and luckily has the same sparsity pattern as the cotangent laplacian, so we can reuse the row and column indices, as well as the diagonal and entry indices for computation. 
The resulting computation is more complex, involving quaternion hammilton products, and is not entirely symmetrical, however the resolution wasn't too bad. Another issue I faced (unresolved) is that we can't take advantage of vectorized writes to global memory when using atomics as they're only implemented for 4 byte primitives. 
The final diagonal computation was also more challenging to implement, involving more variables than the simple reduce for a cotangent laplacian.
To optimize this I first noted that all contributions were coming from adjacent triangles, and so could probably be modelled using a reduce. In particular, the contribution would come from the case where we use the same vertex for both sides of our edge, which after a subtraction will result in zero, so we can therefor assume that all diagonals will be purely real quaternions with a zero imaginary vector.
The real part can be computed as the dot product of the two edges stemming from our vertex, plus the regular real component computed from only our change in mean curvature.
After all this we can model the diagonals as a reduce by vertex triangle adjacency, where at each adjacency pair we transform the two edges containing our vertex into the resulting diagonal. 

```
Tri <- [v0, v1, v2]
find two edges:
e0 <- v1-v0
e1 <- v2-v0

result <- (rho^2 * area / 9) + dot(e0, e1) / (4 * area)
```


</p>
</details>

<details>
<summary><b>Solving the sparse Linear systems</b></summary>
<p>

### Overview

This project involved solving two sparse linear systems. The first called Similarity transforms, produces a quaternion per vertex that best describes it's conformal deformation, using the intrinsic dirac matrix as a left hand side.
The second called Spin positions, produces new vertex positions that best fit the edges calculated using the similarity transforms. 
This was probably the most challenging and important stage of the project. These two processes are the slowest host functions as well, meaning it was crucial to speed them up.
I ended up implementing two methods for both, one using a direct solve via the cuSolverSp and cuSparse API, and the other using an iterative conjugate gradient solver via the CUSP API.

Interestingly the conjugate gradient solve out performs a direct cholesky decomposition drastically for the similarity transforms, and beats the host version for some of the smaller meshes.
The direct solve wins for larger meshes in the Spin positions step, where the iterative solve performs poorly.

</p>
</details>

## Future improvements
I ran out of time before I was able to implement the mean curvature calculation and projection on the device side, and would add this next.
I would also work on moving duplicate code out of the sparse matrix assembly kernels which would allow the shuffle methods to be used by other functions,
such as face area calculations. 
Finally I would like to have used streams more as there is definitely potential to overlap computation in this project.

## Benchmarks
### Host
|                                      |            |             |             |           |   
|--------------------------------------|------------|-------------|-------------|-----------|--- 
|                                      |            |             |             |           |   
| name                                 | iterations | real_time   | cpu_time    | time_unit |   
| HOST_cotangent_laplacian_cube        | 182627     | 3852.71     | 3852.37     | ns        |   
| HOST_cotangent_laplacian_spot        | 90         | 7.68602e+06 | 7.65249e+06 | ns        |   
| HOST_cotangent_laplacian_bunny       | 35         | 2.03471e+07 | 2e+07       | ns        |   
| HOST_cotangent_laplacian_tyra        | 8          | 7.18162e+07 | 6.8106e+07  | ns        |   
| HOST_divergent_edges_cube            | 471392     | 1477.79     | 1477.58     | ns        |   
| HOST_divergent_edges_spot            | 100        | 5.24034e+06 | 5.2395e+06  | ns        |   
| HOST_divergent_edges_bunny           | 50         | 1.39971e+07 | 1.39957e+07 | ns        |   
| HOST_divergent_edges_tyra            | 12         | 5.04251e+07 | 5.04111e+07 | ns        |   
| HOST_face_area_cube                  | 4344953    | 164.346     | 164.308     | ns        |   
| HOST_face_area_spot                  | 2074       | 339835      | 339751      | ns        |   
| HOST_face_area_bunny                 | 723        | 974836      | 974671      | ns        |   
| HOST_face_area_tyra                  | 226        | 3.1432e+06  | 3.14241e+06 | ns        |   
| HOST_intrinsic_dirac_cube            | 44546      | 15991       | 15988.7     | ns        |   
| HOST_intrinsic_dirac_spot            | 8          | 7.53803e+07 | 7.53403e+07 | ns        |   
| HOST_intrinsic_dirac_bunny           | 1          | 6.25203e+08 | 6.25151e+08 | ns        |   
| HOST_intrinsic_dirac_tyra            | 1          | 1.83794e+09 | 1.83772e+09 | ns        |   
| HOST_mean_curvature_cube             | 2044611    | 365.646     | 365.569     | ns        |   
| HOST_mean_curvature_spot             | 1228       | 559729      | 559669      | ns        |   
| HOST_mean_curvature_bunny            | 623        | 1.14741e+06 | 1.14732e+06 | ns        |   
| HOST_mean_curvature_tyra             | 162        | 4.46348e+06 | 4.463e+06   | ns        |   
| HOST_orthonormalize_cube             | 912301     | 768.27      | 768.2       | ns        |   
| HOST_orthonormalize_spot             | 5326       | 135622      | 135611      | ns        |   
| HOST_orthonormalize_bunny            | 2243       | 310662      | 310637      | ns        |   
| HOST_orthonormalize_tyra             | 749        | 934909      | 934832      | ns        |   
| HOST_project_basis_cube              | 1746431    | 397.31      | 397.274     | ns        |   
| HOST_project_basis_spot              | 11059      | 62670       | 62664.8     | ns        |   
| HOST_project_basis_bunny             | 4811       | 142988      | 142976      | ns        |   
| HOST_project_basis_tyra              | 1644       | 419267      | 419232      | ns        |   
| HOST_quaternion_matrix_cube          | 64695      | 10676.5     | 10675.6     | ns        |   
| HOST_quaternion_matrix_spot          | 24         | 2.96319e+07 | 2.96293e+07 | ns        |   
| HOST_quaternion_matrix_bunny         | 12         | 5.74237e+07 | 5.7419e+07  | ns        |   
| HOST_quaternion_matrix_tyra          | 3          | 2.02537e+08 | 2.02519e+08 | ns        |   
| HOST_similarity_xform_cube           | 29636      | 23613.9     | 23611.8     | ns        |   
| HOST_similarity_xform_spot           | 1          | 6.35867e+08 | 6.35806e+08 | ns        |   
| HOST_similarity_xform_bunny          | 1          | 4.63868e+09 | 4.63829e+09 | ns        |   
| HOST_similarity_xform_tyra           | 1          | 8.65426e+09 | 8.65345e+09 | ns        |   
| HOST_spin_positions_cube             | 38234      | 18642.5     | 18641       | ns        |   
| HOST_spin_positions_spot             | 1          | 6.29275e+08 | 6.29218e+08 | ns        |   
| HOST_spin_positions_bunny            | 1          | 4.65188e+09 | 4.65144e+09 | ns        |   
| HOST_spin_positions_tyra             | 1          | 8.61795e+09 | 8.61715e+09 | ns        |   
| HOST_vertex_mass_cube                | 3144441    | 224.885     | 224.867     | ns        |   
| HOST_vertex_mass_spot                | 1726       | 412928      | 412894      | ns        |   
| HOST_vertex_mass_bunny               | 575        | 1.21601e+06 | 1.21591e+06 | ns        |   
| HOST_vertex_mass_tyra                | 180        | 3.92145e+06 | 3.9211e+06  | ns        |   
| HOST_vertex_normals_cube             | 398109     | 1786.48     | 1786.32     | ns        |   
| HOST_vertex_normals_spot             | 370        | 4.03304e+06 | 2.02852e+06 | ns        |   
| HOST_vertex_normals_bunny            | 262        | 5.30947e+06 | 2.79652e+06 | ns        |   
| HOST_vertex_normals_tyra             | 118        | 1.05496e+07 | 6.1032e+06  | ns        |   
| HOST_vertex_triangle_adjacency_cube  | 209312     | 2846.6      | 2811.03     | ns        |   
| HOST_vertex_triangle_adjacency_spot  | 1168       | 696248      | 674157      | ns        |   
| HOST_vertex_triangle_adjacency_bunny | 308        | 2.49694e+06 | 2.46715e+06 | ns        |   
| HOST_vertex_triangle_adjacency_tyra  | 86         | 8.45339e+06 | 8.39191e+06 | ns        |   
| HOST_vertex_vertex_adjacency_cube    | 372804     | 1905.26     | 1901.45     | ns        |   
| HOST_vertex_vertex_adjacency_spot    | 172        | 4.13094e+06 | 4.12159e+06 | ns        |   
| HOST_vertex_vertex_adjacency_bunny   | 55         | 1.28761e+07 | 1.28478e+07 | ns        |   
| HOST_vertex_vertex_adjacency_tyra    | 16         | 4.38669e+07 | 4.37587e+07 | ns        |   

### Device
|                                         |            |             |             |           |   
|-----------------------------------------|------------|-------------|-------------|-----------|---
| name                                    | iterations | real_time   | cpu_time    | time_unit |   
| DEVICE_adjacency_matrix_indices_cube    | 1859       | 341886      | 340272      | ns        |   
| DEVICE_adjacency_matrix_indices_spot    | 952        | 815516      | 811983      | ns        |   
| DEVICE_adjacency_matrix_indices_bunny   | 648        | 1.043e+06   | 1.04007e+06 | ns        |   
| DEVICE_adjacency_matrix_indices_tyra    | 303        | 2.30804e+06 | 2.30536e+06 | ns        |   
| DEVICE_cotangent_laplacian_cube         | 2636       | 266425      | 264868      | ns        |   
| DEVICE_cotangent_laplacian_spot         | 1822       | 389558      | 387782      | ns        |   
| DEVICE_cotangent_laplacian_bunny        | 1180       | 592784      | 590895      | ns        |   
| DEVICE_cotangent_laplacian_tyra         | 250        | 2.78091e+06 | 2.77979e+06 | ns        |   
| DEVICE_divergent_edges_cube             | 2755       | 250017      | 248657      | ns        |   
| DEVICE_divergent_edges_spot             | 1829       | 388904      | 386151      | ns        |   
| DEVICE_divergent_edges_bunny            | 1227       | 573293      | 571326      | ns        |   
| DEVICE_divergent_edges_tyra             | 416        | 1.66145e+06 | 1.66131e+06 | ns        |   
| DEVICE_face_area_cube                   | 174640     | 4013.78     | 4013.42     | ns        |   
| DEVICE_face_area_spot                   | 39931      | 18190.5     | 18189       | ns        |   
| DEVICE_face_area_bunny                  | 10000      | 70404.2     | 70398.4     | ns        |   
| DEVICE_face_area_tyra                   | 10000      | 228716      | 228697      | ns        |   
| DEVICE_intrinsic_dirac_cube             | 11973      | 55211.1     | 55206.5     | ns        |   
| DEVICE_intrinsic_dirac_spot             | 605        | 1.15752e+06 | 1.15693e+06 | ns        |   
| DEVICE_intrinsic_dirac_bunny            | 265        | 2.77881e+06 | 2.76758e+06 | ns        |   
| DEVICE_intrinsic_dirac_tyra             | 58         | 1.16946e+07 | 1.16936e+07 | ns        |   
| DEVICE_quaternion_matrix_cube           | 64005      | 10361       | 10360.2     | ns        |   
| DEVICE_quaternion_matrix_spot           | 1619       | 432756      | 432711      | ns        |   
| DEVICE_quaternion_matrix_bunny          | 825        | 844616      | 844546      | ns        |   
| DEVICE_quaternion_matrix_tyra           | 278        | 2.40125e+06 | 2.40105e+06 | ns        |   
| DEVICE_similarity_xform_direct_cube     | 410        | 1.65968e+06 | 1.65722e+06 | ns        |   
| DEVICE_similarity_xform_direct_spot     | 1          | 3.72199e+09 | 3.72164e+09 | ns        |   
| DEVICE_similarity_xform_direct_bunny    | 1          | 1.48061e+10 | 1.48048e+10 | ns        |   
| DEVICE_similarity_xform_direct_tyra     | 1          | 9.21995e+10 | 9.2191e+10  | ns        |   
| DEVICE_similarity_xform_iterative_cube  | 354        | 1.97964e+06 | 1.97947e+06 | ns        |   
| DEVICE_similarity_xform_iterative_spot  | 1          | 8.37923e+08 | 8.35415e+08 | ns        |   
| DEVICE_similarity_xform_iterative_bunny | 1          | 1.60284e+09 | 1.59807e+09 | ns        |   
| DEVICE_similarity_xform_iterative_tyra  | 1          | 4.12793e+09 | 4.12634e+09 | ns        |   
| DEVICE_spin_positions_direct_spot       | 1          | 3.68252e+09 | 3.68218e+09 | ns        |   
| DEVICE_spin_positions_direct_bunny      | 1          | 1.4813e+10  | 1.48117e+10 | ns        |   
| DEVICE_spin_positions_direct_tyra       | 1          | 9.26087e+10 | 9.25993e+10 | ns        |   
| DEVICE_spin_positions_iterative_spot    | 1          | 3.20449e+09 | 3.19223e+09 | ns        |   
| DEVICE_spin_positions_iterative_bunny   | 1          | 2.43237e+10 | 2.42893e+10 | ns        |   
| DEVICE_spin_positions_iterative_tyra    | 1          | 4.5125e+10  | 4.5104e+10  | ns        |   
| DEVICE_vertex_mass_cube                 | 1850       | 382962      | 381300      | ns        |   
| DEVICE_vertex_mass_spot                 | 774        | 918686      | 914777      | ns        |   
| DEVICE_vertex_mass_bunny                | 715        | 985748      | 981242      | ns        |   
| DEVICE_vertex_mass_tyra                 | 550        | 1.27652e+06 | 1.27083e+06 | ns        |   
| DEVICE_vertex_triangle_adjacency_cube   | 7564       | 83150.4     | 83142.5     | ns        |   
| DEVICE_vertex_triangle_adjacency_spot   | 975        | 645653      | 644782      | ns        |   
| DEVICE_vertex_triangle_adjacency_bunny  | 687        | 951790      | 950354      | ns        |   
| DEVICE_vertex_triangle_adjacency_tyra   | 341        | 2.00665e+06 | 2.00445e+06 | ns        |   
| DEVICE_vertex_vertex_adjacency_cube     | 1754       | 402834      | 401607      | ns        |   
| DEVICE_vertex_vertex_adjacency_spot     | 328        | 2.11945e+06 | 2.11526e+06 | ns        |   
| DEVICE_vertex_vertex_adjacency_bunny    | 205        | 3.42353e+06 | 3.41836e+06 | ns        |   
| DEVICE_vertex_vertex_adjacency_tyra     | 78         | 8.39362e+06 | 8.38729e+06 | ns        |   
