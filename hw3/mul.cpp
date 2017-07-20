#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define ROOT_RANK 0
#define CART_DIM 3

struct grid_info{
    int row_processors;
    int row;
    int col;
    int k;
    int my_rank;
    MPI_Comm comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm c_comm;
    MPI_Comm layer_comm;
};

void free_2d_matrix(double **M, int n){
    int i;
    for (int i = 0; i < n; i++){
        free(M[i]);
    }
    free(M);
}

void generate_rand_matrix(int n, double* &A){
    A = (double *) malloc(n*sizeof(double));

    for(int i = 0; i < n; i++){
        A[i] = drand48();
    }
}

int main(int argc, char *argv[]){
    int n, p, c, rank; 
    int verification;
    
    grid_info grid;

    int ele_per_node;

    double *c_gather;
    double **A, **B, **C;
    
    double start, end;
    MPI_Init(&argc, &argv);
     
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grid.my_rank); 
    // Create cartesian topology 
    
    n = atoi(argv[1]);
    c = atoi(argv[2]); 
    verification = atoi(argv[3]);
    grid.row_processors = (int)sqrt(p/c);
    ele_per_node = n*n*c/(p);

    int dims[3] = {grid.row_processors, grid.row_processors, c};
    int periods[3] = {1,1,1};
    int coords[3];

    MPI_Cart_create(MPI_COMM_WORLD, CART_DIM, dims, periods, 1, &grid.comm); 
    MPI_Cart_coords(grid.comm, grid.my_rank, CART_DIM, coords);
    
    grid.row = coords[0];
    grid.col = coords[1];
    grid.k = coords[2];
    int xy_cart_sub_dim[3] = {1,1,0};
    MPI_Cart_sub(grid.comm,xy_cart_sub_dim, &(grid.layer_comm));

    int z_cart_sub_dim[3] = {0,0,1};
    MPI_Cart_sub(grid.comm, z_cart_sub_dim, &(grid.c_comm));

    int row_cart_sub_dim[3] = {0,1,0};
    MPI_Cart_sub(grid.comm, row_cart_sub_dim, &(grid.row_comm)); 

    int col_cart_sub_dim[3] = {1,0,0};
    MPI_Cart_sub(grid.comm, col_cart_sub_dim, &(grid.col_comm));

    // Create Matrix a and b on rank 0
     
    MPI_Barrier(grid.col_comm);
    start = MPI_Wtime();
    double *a, *b;
    if(grid.my_rank == ROOT_RANK){
        srand48(time(0));
        generate_rand_matrix(n*n, a);
        generate_rand_matrix(n*n, b);
    }

    //scatter the sub matrices over (1,1,0) plane
    double *local_a = (double *) malloc(ele_per_node * sizeof(double));
    double *local_b = (double *) malloc(ele_per_node * sizeof(double));
    double *local_a_tmp = (double *) malloc(ele_per_node * sizeof(double));
    double *local_b_tmp = (double *) malloc(ele_per_node * sizeof(double));
    double * local_as[2] = {local_a, local_a_tmp};// support non-blocking send/recv: one buffer is used for sending data while the other for computation(and fetching data from another node); the two bufferes switch responsibility in next iteration.
    double * local_bs[2] = {local_b, local_b_tmp};

    if(grid.k == 0){
    MPI_Scatter(a, ele_per_node, MPI_DOUBLE, local_a ,ele_per_node, MPI_DOUBLE, ROOT_RANK, grid.layer_comm); 
    MPI_Scatter(b, ele_per_node, MPI_DOUBLE, local_b, ele_per_node, MPI_DOUBLE, ROOT_RANK, grid.layer_comm);
    }
    
    MPI_Barrier(grid.layer_comm);
    //Broadcast the sub matrices along (0,0,1) direction
    MPI_Bcast(local_a, ele_per_node, MPI_DOUBLE, 0, grid.c_comm);
    MPI_Bcast(local_b, ele_per_node, MPI_DOUBLE, 0, grid.c_comm);

    MPI_Barrier(grid.c_comm);

    //Generate initial config.
    MPI_Status status;
    int source_a, dest_a;
    MPI_Cart_shift(grid.row_comm, 0, -grid.row + grid.k*(grid.row_processors/c), &source_a, &dest_a);
    int recv = 0;
    int send = 0;
    MPI_Request request_a_send, request_a_recv, request_b_send, request_b_recv;
    MPI_Isend(local_as[0], ele_per_node, MPI_DOUBLE, dest_a, send, grid.row_comm, &request_a_send);
    MPI_Irecv(local_as[1], ele_per_node, MPI_DOUBLE, source_a, recv, grid.row_comm, &request_a_recv);

    int source_b, dest_b;
    MPI_Cart_shift(grid.col_comm, 0, -grid.col + grid.k*(grid.row_processors/c), &source_b, &dest_b);
    MPI_Isend(local_bs[0], ele_per_node, MPI_DOUBLE, dest_b, send, grid.col_comm, &request_b_send);
    MPI_Irecv(local_bs[1], ele_per_node, MPI_DOUBLE, source_b, recv, grid.col_comm, &request_b_recv);
    
    
    double *c_local = (double *) malloc(ele_per_node *sizeof(double));

    int local_upper_bound = n/grid.row_processors;
    int idx = 0;
    for(int i = 0; i < local_upper_bound; i++){
        for (int j = 0; j < local_upper_bound; j++){
            c_local[idx] = 0;
            idx++;
        }
    }

    MPI_Wait(&request_b_send, &status);
    MPI_Wait(&request_b_recv, &status);
    MPI_Wait(&request_a_send, &status);
    MPI_Wait(&request_a_recv, &status);
    int source_row, dest_row;
    MPI_Cart_shift(grid.row_comm, 0, 1, &source_row, &dest_row);

    int source_col, dest_col;
    MPI_Cart_shift(grid.col_comm, 0, 1, &source_col, &dest_col);

    //perform cannon's algorithm
    int alternate = 1;
    for(int i_outer = 1; i_outer < grid.row_processors/c; i_outer++){
        
        MPI_Isend(local_as[alternate], ele_per_node, MPI_DOUBLE, dest_row, send, grid.row_comm, &request_a_send);
        MPI_Irecv(local_as[1-alternate], ele_per_node, MPI_DOUBLE, source_row, recv, grid.row_comm, &request_a_recv);
        MPI_Isend(local_bs[alternate], ele_per_node, MPI_DOUBLE, dest_col, send, grid.col_comm, &request_b_send);
        MPI_Irecv(local_bs[1-alternate], ele_per_node, MPI_DOUBLE, source_col, recv, grid.col_comm, &request_b_recv);

        int idx = 0;
        for(int i = 0; i < local_upper_bound; i++){
            for (int j = 0; j < local_upper_bound; j++){
                for(int k = 0; k < local_upper_bound; k++){
                    c_local[idx] += local_as[alternate][i*local_upper_bound + k] * local_bs[alternate][k*local_upper_bound + j];
                }
                idx++;
            }
        }
        alternate = 1 - alternate;//sending buffer becomes computation buffer ; computation buffer becomes sending buffer in next iteration  
        MPI_Wait(&request_b_send, &status);
        MPI_Wait(&request_b_recv, &status);
        MPI_Wait(&request_a_send, &status);
        MPI_Wait(&request_a_recv, &status);

    }

    idx = 0;
    for(int i = 0; i < local_upper_bound; i++){
        for (int j = 0; j < local_upper_bound; j++){
            for(int k = 0; k < local_upper_bound; k++){
                c_local[idx] += local_as[alternate][i*local_upper_bound + k] * local_bs[alternate][k*local_upper_bound + j];
            }
            idx++;
        }
    }

    //Gather the results on root node
    double * gather_c =  (double *) malloc(ele_per_node*sizeof(double));
    MPI_Reduce(c_local, gather_c, ele_per_node, MPI_DOUBLE, MPI_SUM, 0, grid.c_comm);
    free(c_local);
    free(local_a);
    free(local_b);

    double * c_all;
    if(grid.my_rank == ROOT_RANK){
        c_all = (double *) malloc(n * n* sizeof(double));
    }

    MPI_Barrier(grid.comm);
    if(grid.k == 0){
        MPI_Gather(gather_c, ele_per_node, MPI_DOUBLE, c_all, ele_per_node, MPI_DOUBLE, ROOT_RANK, grid.layer_comm);

    }
    MPI_Barrier(grid.comm);
    end = MPI_Wtime();
    free(gather_c);
    
    //Recover A, B, C
    if(grid.my_rank == ROOT_RANK){
         
        A = (double **) malloc(n*sizeof(double *));
        B = (double **) malloc(n*sizeof(double *));
        C = (double **) malloc(n*sizeof(double *));
        for (int i = 0; i < n ; i++){
            A[i] = (double *) malloc(n * sizeof(double));
            B[i] = (double *) malloc(n * sizeof(double));
            C[i] = (double *) malloc(n * sizeof(double));
        }
        for(int i = 0; i < grid.row_processors; i++){
            for (int j = 0; j < grid.row_processors; j++){
                int pid = i * grid.row_processors + j;
                for (int row = 0; row < local_upper_bound; row++){
                    int row_idx = i * local_upper_bound + row;
                    for(int col = 0; col < local_upper_bound; col++){
                        int col_idx = j * local_upper_bound + col;
                        int local_idx = pid * local_upper_bound * local_upper_bound + row * local_upper_bound + col; 
                        A[row_idx][col_idx] = a[local_idx];
                        B[row_idx][col_idx] = b[local_idx];
                        C[row_idx][col_idx] = c_all[local_idx];
                    }
                }
            }
        }
        
        //Verification
        if(verification){
        double res = 0;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                double tmp = 0;
                for (int k = 0; k < n; k++){
                    tmp +=  A[i][k] * B[k][j];
                }
                res += fabs(tmp-C[i][j]);
            }
        }
        res /= (n*n);
        std::cout << "residual " << res << std::endl;
        }
        //std::cout << "n: " << n << " "<< "p: " << p <<  " " << "c: " << c << std::endl; 
        std::cout << "MPI time spent " << (end - start) << std::endl;
        free_2d_matrix(A, n);
        free_2d_matrix(B, n);
        free_2d_matrix(C, n);
        free(c_all);
    } 
    MPI_Finalize();
}
