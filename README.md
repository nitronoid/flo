# Conformal Willmore flow
Based on the work of Keenan Crane, this is an implementation of Conformal curvature flow using spin transformations.

## Dependencies
- Eigen
- igl
- Suite-Sparse (cholmod)
- qmake
- gsl-lite (git submodule)

## Build
```
> git submodule update --init --recursive
> qmake
> make -j
```

## References:
    [1] K. Crane, U. Pinkall, and P. Schröder, “Spin transformations of discrete surfaces,” ACM Transactions on Graphics, vol. 30, no. 4, p. 1, Jul. 2011.
        
    [2] K. Crane, U. Pinkall, and P. Schröder, “Robust fairing via conformal curvature flow,” ACM Transactions on Graphics, vol. 32, no. 4, p. 1, Jul. 2013.
                
