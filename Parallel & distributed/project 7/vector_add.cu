/*
	Mkhanyisi Gamedze
	CS336 Prallel & Distributed processing
	project 7
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_THREADS_PER_BLOCK 1025
#define NUM_BLOCKS 1
#define N (NUM_THREADS_PER_BLOCK*NUM_BLOCKS)

__global__ 
void add( float *a, float *b, float *c ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
int run_main( void ) {
	
    float a[N], b[N], c[N];
    float *dev_a, *dev_b, *dev_c;
	float time;
	cudaEvent_t start, stop;
	
    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N * sizeof(float) ) ;
    cudaMalloc( (void**)&dev_b, N * sizeof(float) ) ;
    cudaMalloc( (void**)&dev_c, N * sizeof(float) ) ;

    // fill the arrays 'a' and 'b' on the CPU
    int i;
    for (i=0; i<N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a,a, N * sizeof(float),cudaMemcpyHostToDevice ) ;
    cudaMemcpy( dev_b,b,N * sizeof(float),cudaMemcpyHostToDevice ) ;
    
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    add<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>( dev_a, dev_b, dev_c );
    
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( c,dev_c,N * sizeof(float),cudaMemcpyDeviceToHost ) ;
    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error: %f + %f != %f\n", a[i], b[i], c[i] );
            success = false;
        }
    }
    if (success) printf( "We did it!\n" );
    printf("\nRun time:  %.6f ms \n\n", time);

    // free the memory allocated on the GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
	
    return 0;
}

int main( void ) {
	run_main();
	run_main();
	run_main();
	run_main();
	run_main();
	run_main();
}
