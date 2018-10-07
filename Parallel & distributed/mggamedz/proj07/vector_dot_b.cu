/*
	Mkhanyisi Gamedze
	CS336 Prallel & Distributed processing
	project 7
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*   tree version  strategy for summing the products */

#define imin(a,b) (a<b?a:b)
#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)

const int N = 33 * 1024;
const int threadsPerBlock = 8192;
const int blocksPerGrid =imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );

// device dot product function
__global__ 
void dot( float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main( void ) {
    float   *a, *b, c, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;
    float time;
	cudaEvent_t start, stop;

    // allocate memory on the CPU side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a,N*sizeof(float) ) ;
    cudaMalloc( (void**)&dev_b,N*sizeof(float) ) ;
    cudaMalloc( (void**)&dev_partial_c,blocksPerGrid*sizeof(float) ) ;

    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N*sizeof(float),cudaMemcpyHostToDevice );
    
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    
    // compute dot product
    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,dev_partial_c );
    
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( partial_c, dev_partial_c,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost );

    // finish up on the CPU side
    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }
	
	printf("\nTime naive method:  %.6f ms \n\n", time);
   
    printf( "GPU value %.6g  Expected value %.6g\n", c,2 * sum_squares( (float)(N - 1) ) );

    // free memory on the GPU side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_partial_c );

    // free memory on the CPU side
    free( a );
    free( b );
    free( partial_c );
}