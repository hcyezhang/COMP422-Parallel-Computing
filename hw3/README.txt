Matrix multiplication is a binary operation performed on a pair of matrices A, rank M x N, and B, rank N x P, resulting in a matrix C, rank M x P. In matrix multiplication, the element-wise products from a row of elements from A, and a column of elements from B are summed to generate an element of matrix C.

This program is an implementation of 2.5D version of matrix multiplication.

Summary of the 2.5D algorithm to get you started: 2.5D matrix multiplication algorithm, originating from UC Berkeley, reduces communication bandwidth consumed and exposed latency in matrix multiplication by maintaining a total of 'C' replicas of the input matrices A and B on a 3D grid of processors, P in total, of dimension sqrt(P/C) x sqrt(P/C) x C. Initially, a single copy of the A, B matrices is broken into chunks, and distributed among the processors on the front face of this grid. Through a series of broadcasts, and point-to-point communications, these chunks are then distributed to different processors. Each processor in the grid performs a matrix multiplication on the chunks of A and B received, and a final phase of reduction produces the result, C, matrix on the front face of the grid.

Please refer to the technical report entitled “Communication-optimal parallel 2.5D matrix multiplication and LU factorization algorithms” by Solomonik et. al. for complete information on this algorithm.

———————————————————————————————————————————————————————————————-

To run the program, please do the following:
moduel load OpenMPI
make
srun -n<number of nodes> mul <size of square matrix> <c> <verification>

i.e. srun -n 32 mul 7500 2 0 if you would like to test the performance of 7500 by 7500 matrix multiplication on 32 cores.

srun -n 32 mul 1000 2 1 if you would like to verify the correctness of implementation.
