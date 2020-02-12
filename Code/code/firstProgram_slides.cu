#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "utils.h"

using std::vector;

__device__ __host__
int f(int i) {
    return i*i;
}

__global__
void kernel(int* out) {
    out[threadIdx.x] = f(threadIdx.x);
}

int main(int argc, char** argv) {
    int N = 32;

    if(argc == 2) {
        N = atoi(argv[1]);
    }

    int* d_output;
    cudaMalloc(&d_output, sizeof(int) * N);
    kernel<<<1, N>>>(d_output);
    vector<int> h_output(N);
    cudaMemcpy(&h_output[0], d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i)
		printf("Entry %3d, written by thread %2d\n", h_output[i], i);
    cudaFree(d_output);

    return 0;
}

#if 0
int* d_output;
cudaMalloc(&d_output, sizeof(int) * N);
kernel<<<1, N>>>(d_output);
vector<int> h_output(N);
cudaMemcpy(&h_output[0], d_output, sizeof(int) * N,
           cudaMemcpyDeviceToHost);
for(int i = 0; i < N; ++i) {
   printf("Entry %3d, written by thread %2d\n",
          h_output[i], i);
}
cudaFree(d_output);
#endif
