#ifdef OPENMP
# include <omp.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <sys/time.h>

#define SCALE_FACTOR 5000 
typedef struct Compare { double val; int index; } Compare;
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)

void generate_rand_matrix(int num_threads, int n, double** &A, double** &A_copy){
    A = (double **) malloc(n*sizeof(double *));
    A_copy = (double **) malloc(n*sizeof(double *));
   
    for (int i = 0; i< n; i ++){ 
        A[i] = (double*)malloc(n*sizeof(double));
        A_copy[i] = (double*)malloc(n*sizeof(double));
    }
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = drand48()*SCALE_FACTOR;
        }
    }

    for (int i = 0; i < n; i++) std::memcpy(A_copy[i], A[i], n*sizeof(double));
}
    
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

int decomposition(int n, double** A, double** &L, double** &U, int* &P){
    
    P = (int*)malloc(n*sizeof(int));
    L = (double **) malloc(n*sizeof(double *));
    U = (double **) malloc(n*sizeof(double *));
     
    Compare max;
    max.val = 0;
    max.index = 0;
    #pragma omp parallel shared(L,U,A)      
    {
    
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
        L[i] = (double*)malloc(n*sizeof(double));
        U[i] = (double*)malloc(n*sizeof(double));

    }
    
    
    
    #pragma omp for schedule(static) reduction(maximum: max)
    for (int i = 0; i < n; i++){
        P[i] = i;
        memset(U[i], n*sizeof(double), 0);
        memset(L[i], n*sizeof(double), 0);
        L[i][i] = 1;
        
        if(abs(A[i][0]) > max.val){
            max.val = abs(A[i][0]);
            max.index = i;
        }

    }
    
    }
    double tmp_p;
    double tmp_a[n];
    double tmp_l[n];
    #pragma omp parallel shared(L,U,A)
    {
        for (int k = 0; k < n; k++){
    #pragma omp master
        {
            int k_swap = max.index;
             
            tmp_p = P[k];
            P[k] = P[k_swap];
            P[k_swap] = tmp_p;

                        
            std::memcpy(tmp_l,L[k], k*sizeof(double));
            std::memcpy(L[k],L[k_swap], k*sizeof(double));
            std::memcpy(L[k_swap], tmp_l, k*sizeof(double));
            
            std::memcpy(U[k]+k,A[k_swap]+k, (n-k)*sizeof(double));
            std::memcpy(A[k_swap]+k, A[k]+k, (n-k)*sizeof(double));

            max.val = 0.;
            max.index = k;

            }
            
    #pragma omp barrier    
    #pragma omp for schedule(static) reduction(maximum:max) 
            for (int i = k+1; i < n; i++){
                L[i][k] = A[i][k]/U[k][k];
                for (int j = k+1; j < n; j++){
                    A[i][j] = A[i][j] - L[i][k]*U[k][j];
                }

                if(abs(A[i][k+1]) > max.val){
                    max.val = abs(A[i][k+1]);
                    max.index = i;
                }
                if(max.val == 0.) std::cout << "Singular Matrix" << std::endl;
            }
        }
    }
        return 0;  
}                

int main(int argc, const char* argv[]){
   int n, num_threads, vf;
   n = atoi(argv[1]);
   num_threads = atoi(argv[2]);
   //omp_set_num_threads(num_threads);
   vf = atoi(argv[3]);
   
   double** A_copy;
   double** A;
   generate_rand_matrix(num_threads,n, A, A_copy);
   double** L;
   double** U;
   int* P;
   
   struct timeval start, end;
   long double diff;
   gettimeofday(&start,NULL);
   
   decomposition(n,A,L,U,P);
   gettimeofday(&end, NULL);
   
   diff = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec;
   
   if(vf){
        double res = verification(n,A_copy,L,U,P);
        std::cout << res << std::endl;
   }
   std::cout << diff << std::endl;

   return 0;

}
