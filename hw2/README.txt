LU Decomposition (where 'LU' stands for 'lower upper') is a classical method for transforming an N x N matrix A into the product of a lower-triangular matrix L and an upper-triangular matrix U,

A = LU.

The two shared-memory parallel programs perform LU decomposition using Gaussian elimination with row pivoting. One solution uses OpenMP with work-sharing constructs and one using OpenMP strictly without work-sharing constructs. Each LU decomposition implementation should accept two arguments: n - the size of a matrix, followed by t - the number of threads. Your programs will allocate an n x n matrix a of double precision (64-bit) floating point variables.  

————————————————————————————————————————————

I have commented lu-non-worksharing.cpp in detail(I did not comment the other two as most parts are exactly the same).
Each program takes three arguments, the dimention of matrix A, the number of threads and verification(0 means no verification).
To run the program, i.e. ./xxxx 8000 32 0
To check correctness, i.e. ./xxxx 1000 8 1