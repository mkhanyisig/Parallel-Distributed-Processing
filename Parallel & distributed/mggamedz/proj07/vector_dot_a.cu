/*
	Mkhanyisi Gamedze
	CS336 Prallel & Distributed processing
	project 7
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* "naive" strategy for summing the products  */

#define imin(a,b) (a<b?a:b)
#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)

const int N = 33 * 1024;
const int threadsPerBlock = 8192;
const int blocksPerGrid =imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );


__global__ 
void dotProd_naive(float *a, float *b, float *result){
  /*
    The naïve way to accomplish this reduction 
    one thread iterate over the shared memory and calculate a running sum
  */
	
  __shared__ float cache[threadsPerBlock];
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int cacheIndex = threadIdx.x;
  float temp = 0;
  while (idx < N) {
    temp += a[idx] * b[idx];
    idx += threadsPerBlock; //skip to next block
  }
  
  cache[cacheIndex] = temp;
  
  //clear temp:
  temp=0;

  __syncthreads();  // wait for other threads to complete
  
  //if first thread (only thread 1) 
  if(blockIdx.x*blockDim.x+threadIdx.x==0){
    //calculate the result:
    int i;
    for(i=0;i<threadsPerBlock;i++){
      temp+=cache[i]; 
    }
    result[0]=temp;
  }
}


int main( void ) {
    float   *a, *b, c, *result_h;
    float   *dev_a, *dev_b, *result_d;
    float time;
	cudaEvent_t start, stop;

    // allocate memory on the CPU side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    result_h = (float*)malloc( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a,N*sizeof(float) ) ;
    cudaMalloc( (void**)&dev_b,N*sizeof(float) ) ;
    cudaMalloc( (void**)&result_d,sizeof(float) ) ;

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
    dotProd_naive<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,result_d );
    
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( result_h, result_d,sizeof(float),cudaMemcpyDeviceToHost );

    // finish up on the CPU side
    c = result_h[0];
   
	printf("\nTime naive strategy:  %.6f ms \n\n", time);
    
    printf( "GPU value %.6g  Expected value %.6g\n", c,2 * sum_squares( (float)(N - 1) ) );

    // free memory on the GPU side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( result_d );

    // free memory on the CPU side
    free( a );
    free( b );
    free( result_h );
}