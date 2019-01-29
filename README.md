# Conformal Willmore flow
Based on the work of Keenan Crane, this is an implementation of Conformal curvature flow using spin transformations.

## Dependencies
- Eigen (header only, version 3.3.7)
- igl (header only)
- Suite-Sparse (cholmod)
- qmake (version 5)
- gsl-lite (header only, git submodule)

#### Dependencies (Device code only)
- CUDA (version 9.0 or greater)
- Thrust (shipped with CUDA)

## Build
```
> git submodule update --init --recursive
> qmake
> make -j
```

## References:
    [1] K. Crane, U. Pinkall, and P. Schröder, “Spin transformations of discrete surfaces,” ACM Transactions on Graphics, vol. 30, no. 4, p. 1, Jul. 2011.
        
    [2] K. Crane, U. Pinkall, and P. Schröder, “Robust fairing via conformal curvature flow,” ACM Transactions on Graphics, vol. 32, no. 4, p. 1, Jul. 2013.
                

## Development of parallel implementation

<details>
<summary><b>Vertex triangle adjacency</b></summary>
<p>

### Overview

Vertex triangle adjacency operates on a list of integer 3 tuples, where each integer represents one of the vertices in the triangular face.<br>
From this list we calculate and return an adjacency list, that is a list of the faces which one vertex neighbours.<br>
We also return a vertex-face valence (the number of neighbouring faces), and a cumulative version of the valence starting at 0.<br>
The cumulative valence is used as an offset into the adjacency list, i.e. the faces that neighbour vertex I span the range 

```
adjacency_list[cumulative_valence[I]] --- adjacency_list[cumulative_valence[I] + valence[I]]
```

### Implementation

This process can be implemented as a histogram. Each vertex acts as a bucket, and we want to count the amount of faces in each bucket.<br>
I tested several different methods for computing the three output lists:

#### Simulataneous sort method

The first process I implemented used the thrust library histogram example. The algorithm was:

+ Generate ascending list of faces id's corresponding to the vertex id's == [0,0,0, 1,1,1, ..., n,n,n], where n is the number of faces*3
+ Simulataneously sort the vertex and generated face id's, using the vertex id's as the sort key
+ Calculate a cumulative histogram from the sorted vertex id's by finding the last occurence of every unique id (upper_bound)
+ Calculate the final histogram by subtracting every left neighbour in the cumulative list from it's immediate right neighbour (adjacent_difference)

The simulataneous sort can be implemented using a zip_iterator, and a sort by key:

```cpp
auto ptr_tuple = thrust::make_tuple(vert_idx.begin(), face_idx.begin());
auto zip_begin = thrust::make_zip_iterator(ptr_tuple);
thrust::sort_by_key(vert_idx.begin(), vert_idx.end(), zip_begin);
```

The valence and cumulative valence are then calculated as histograms using:

```cpp
thrust::counting_iterator<int> count(0);
thrust::upper_bound(vert_idx.begin(), vert_idx.end(), count, count + nbins, cumulative_valence.begin());
thrust::adjacent_difference(cumulative_valence.begin()+1, cumulative_valence.end(), valence.begin());
```

After implementing this method I conducted some benchmarks and found the timings to be disapointing.<br>
I then benchmarked each stage of the algorithm individually:<br>
As we can see, 90% of the run-time was spent on the sort_by_key function. <br>
I then conducted further benchmarks into variations of the thrust sorting api, and as the results show,<br>
the algorithm performs poorly when using the zip_iterator.

This lead me to investigate alternative methods without the zip_iterator.


#### Gather method
This process is similar to the previous method, however we no longer conduct a simultaneous sort.<br>
Instead we sort an ascending list of integers [0 - n], where n is number of vertex ids.<br>
We then use the thrust gather function to reorder both the vertex ids and face ids. <br>

```cpp
thrust::sequence(indices.begin(), indices.end());
thrust::sort_by_key(vert_idx.begin(), vert_idx.end(), indices.begin());
thrust::gather(indices.begin(), indices.end(), vert_idx.begin(), vert_idx.begin());
thrust::gather(indices.begin(), indices.end(), face_idx.begin(), face_idx.begin());
```

This produces faster timings. The gather operations are quick, and the sort_by_key performs much better with a simple array to reorder.<br>

Next I wanted to compare these results with the simple atomic counter method.

#### Atomics method
The atomic method has an advantage in that the input data need not be sorted to calculate the histogram.<br>

+ First we count the valence of a vertex using atomicAdd.
+ We then use a prefix_sum to calculate the cumulative valence.
+ Finally we sort the face ids using the vertex ids as the key.

This method only requires us to reorder one array of ids, so our previous sort and gather becomes:<br>

```cpp
thrust::sort_by_key(vert_idx.begin(), vert_idx.end(), face_idx.begin());
```

Benchmarking this method shows the best speed-ups:<br>


</p>
</details>

