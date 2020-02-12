#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include "utils.h"

using std::vector;

int n, n_thread;

void ProcessOpt(int argc, char** argv) {
    int c;
    n = 1024;
    n_thread = 512;

    while((c = getopt(argc, argv, "n:t:")) != -1)
        switch(c) {
            case 'n':
                n = atoi(optarg);
                break;

            case 't':
                n_thread = atoi(optarg);
                break;

            case '?':
                break;
        }
}

__global__
void Initialize(int n, int* a, int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < n) {
        a[n*i + j] = j;
        b[n*i + j] = i-2*j;
    }
}

__global__
void Add(int n, int* a, int* b, int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < n) {
        c[n*i + j] = a[n*i + j] + b[n*i + j];
    }
}

int main(int argc, char** argv) {

    ProcessOpt(argc,argv);

    printf("Dimensions of matrix: %5d x %5d\n",n,n);

    int* d_a, *d_b, *d_c;

    /* Allocate memory */
    checkCudaErrors(cudaMalloc(&d_a, sizeof(int) * n*n));
    checkCudaErrors(cudaMalloc(&d_b, sizeof(int) * n*n));
    checkCudaErrors(cudaMalloc(&d_c, sizeof(int) * n*n));

    dim3 threads_per_block(2,n_thread);
    int blocks_per_grid_x = (n + 2 - 1) / 2;
    int blocks_per_grid_y = (n + n_thread - 1) / n_thread;
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);
    Initialize<<<num_blocks, threads_per_block>>>(n, d_a, d_b);
    Add<<<num_blocks, threads_per_block>>>(n, d_a, d_b, d_c);

    /* Note that kernels execute asynchronously.
       They will fail without any error message!
       This can be confusing when debugging.
       The output arrays will be left uninitialized with no warning.
       */

    vector<int> h_c(n*n);
    /* Copy the result back */
    checkCudaErrors(cudaMemcpy(&h_c[0], d_c, sizeof(int) * n*n,
                               cudaMemcpyDeviceToHost));

    /* Test result */
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(!(h_c[n*i + j] == i-j)) {
                printf("%d %d %d %d %d\n",n,i,j,h_c[n*i + j],i-j);
            }

            assert(h_c[n*i + j] == i-j);
        }
    }

    printf("All tests have passed; calculation is correct.\n");

    return 0;
}
