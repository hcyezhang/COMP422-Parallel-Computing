#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <cstring>
#include <sys/time.h>
#include <numa.h>

#define SCALE_FACTOR 5000

/*generate_rand_matrix generates a random square matrix A with given size n*/
void generate_rand_matrix(int num_threads, int n, double** &A, double** &A_copy){
    A = (double **) malloc(n*sizeof(double *));
    A_copy = (double **) malloc(n*sizeof(double *)); //A_copy is used for verification

#pragma omp parallel num_threads(num_threads) 
{
  int tid = omp_get_thread_num();
    for (int i = tid; i< n; i += num_threads){ //thread-data mapping
        A[i] = (double*)malloc(n*sizeof(double));
        A_copy[i] = (double*)malloc(n*sizeof(double));
    }
}

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = drand48()*SCALE_FACTOR;
        }
    }

    for (int i = 0; i < n; i++) std::memcpy(A_copy[i], A[i], n*sizeof(double));
}

/*verification computes the L2-norm of |A-LU|*/
double verification(int n, double** A, double** L, double** U, int* P){
    double res = .0;
    for (int k = 0; k < n; k++){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                A[P[k]][i] -= L[k][j]*U[j][i];
            }

        res += A[P[k]][i] * A[P[k]][i];
        }
    }
    return res;
}

int decomposition(int num_threads, int n, double** A, double** &L, double** &U, int* &P, double** &A_copy){

  P = (int*)malloc(n*sizeof(int));
  L = (double **) malloc(n*sizeof(double *));
  U = (double **) malloc(n*sizeof(double *));

  int max = 0;
  int k_swap = 0;
  int k_back = k_swap;

  double tmp_l[n];

/*Initialization Stage*/
#pragma omp parallel num_threads(num_threads) shared(L,U,A_copy,A,k_swap,max)
  {
    int tid = omp_get_thread_num();
    double local_max = 0;
    int local_k = 0;

    for (int i = tid; i < n; i += num_threads) { //thread-data mapping
      L[i] = (double*)malloc(n*sizeof(double));
      U[i] = (double*)malloc(n*sizeof(double));
      memset(U[i], n*sizeof(double), 0);
      memset(L[i], n*sizeof(double), 0);
      P[i] = i;
      L[i][i] = 1;
      if(abs(A[i][0]) > local_max){ // each thread computes a local_max
        local_max = abs(A[i][0]);
        local_k = i;
      }

    }
#pragma omp critical
    {
      if(local_max > max){ // all threads then enter this critical region to compute a global_max
        max = local_max;
        k_swap = local_k;
      }
    }

#pragma omp barrier

/*Computing LU decomposition*/
    for (int k = 0; k < n; k++){
      if(k%num_threads == tid) // the thread which owes the kth row is selected to perform the swapping operation.
      {
        int tmp_p = P[k];
        P[k] = P[k_swap];
        P[k_swap] = tmp_p;

        std::memcpy(tmp_l,L[k], k*sizeof(double));
        std::memcpy(L[k],L[k_swap], k*sizeof(double));
        
        std::memcpy(U[k]+k,A[k_swap]+k, (n-k)*sizeof(double));
        std::memcpy(A[k_swap]+k, A[k]+k, (n-k)*sizeof(double));//A[k_swap] resides on U, no need to care about the previous k rows of A. [0:k-1] of the row has been cancelled out to zero.
        
        k_back = k_swap; //a trivial optimization - record k_swap for this round (used in the last loop)
        max = 0.;
        k_swap = k;
      }

#pragma omp barrier
      double local_max = 0;
      int local_k = k + 1;
      int start_idx = (local_k)%num_threads;//num_threads is the period
      if(tid - start_idx >= 0){//(tid-start_idx) is the offset
        start_idx = local_k + tid - start_idx;
      }
      else{
        start_idx = local_k + tid - start_idx + num_threads;
      }

      for (int i = start_idx; i < n; i += num_threads){
        L[i][k] = A[i][k]/U[k][k];
        for (int j = k+1; j < n; j++){
          A[i][j] = A[i][j] - L[i][k]*U[k][j];
        }
        if(abs(A[i][k+1]) > local_max){ // compute max for next round
          local_max = abs(A[i][k+1]);
          local_k = i;
        }
      }
#pragma omp critical
      {
        if(local_max > max){
          max = local_max;
          k_swap = local_k;
        }
      }
      if(k_back%num_threads == tid)// the thread corresponds to k_back can write data faster
      {
        std::memcpy(L[k_back], tmp_l, k*sizeof(double));
      }
#pragma omp barrier

    }
}
return 0;
}

int main(int argc, const char* argv[]){
   int n, num_threads, vf;
   n = atoi(argv[1]);
   num_threads = atoi(argv[2]);
   omp_set_num_threads(num_threads);
   vf = atoi(argv[3]);
   
   double** A;
   double** A_copy;
   generate_rand_matrix(num_threads, n, A, A_copy);
   double** L;
   double** U;
   int* P;
   
   /***timing***/
   struct timeval start, end;
   long double diff;
   gettimeofday(&start,NULL);
   decomposition(num_threads,n,A,L,U,P,A_copy);
   gettimeofday(&end, NULL);
   diff = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec;

   if(vf){
        double res = verification(n,A_copy,L,U,P);
        std::cout << res << std::endl;
   }
   std::cout << diff << std::endl;

   return 0;

}
