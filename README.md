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