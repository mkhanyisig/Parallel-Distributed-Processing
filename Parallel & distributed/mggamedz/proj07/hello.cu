#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREADS 5
#define NUM_BLOCKS 4

// The CPU calls this function, but it's executed on the GPU.
// Use "__device__" if from device.
// Notice that we now add an argument that points to where the vector is stored.
__global__ void kernel_hello(float* A) {
	//printf("Hello from the device. Thread %d, block %d.\n", threadIdx.x, blockIdx.x);
	printf("A[%d] = %f \n", threadIdx.x + blockIdx.x * blockDim.x, 
						  A[threadIdx.x + blockIdx.x * blockDim.x] );
	A[threadIdx.x + blockIdx.x * blockDim.x] = 5.0;
}

int main(void) {

    /*
	// Device code -- runs on GPU
	// Host code -- runs on CPU
	cudaDeviceProp prop; // Cuda device property type
	cudaGetDeviceProperties( &prop, 0 ); // Fills in the struct with properties
	
	printf( "named %d\n", prop.name );
	printf( "max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor );
	printf( "number of multiprocs: %d\n", prop.multiProcessorCount );
	printf( "Compute Capability: %d.%d\n", prop.major, prop.minor );
	*/

	printf("Hello from the host\n");
	
	// Making the vector on the host
	float* host_A = (float*)malloc(sizeof(float) * NUM_THREADS * NUM_BLOCKS);
	int i;
	for (i = 0; i < NUM_THREADS * NUM_BLOCKS; i++) {
		host_A[i] = 2*i;
	}
	
	// Allocate space for the vector on the device
	float* device_A = NULL;
	// Do frequent error-checking... The most thorough way to do error check and print error message.
	cudaError_t err; // A CUDA_Error-type thing
	err = cudaMalloc( (void**) &device_A, sizeof(float) * NUM_THREADS * NUM_BLOCKS ); // Takes in address, and how much to malloc
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code is %s), boogbye.\n", cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
	
	// Copy the vector from host to device
	err = cudaMemcpy( device_A, host_A, sizeof(float) * NUM_THREADS * NUM_BLOCKS, 
						cudaMemcpyHostToDevice ); 
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy to device vector A (error code is %s), doogdye.\n", cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
	
	// Tell the function to use this newly malloced slot
	kernel_hello<<< NUM_BLOCKS, NUM_THREADS >>>(device_A); 
	cudaDeviceSynchronize(); // Wait for the device to finish
	
	// After finished, copy the data from device to host.
	err = cudaMemcpy( host_A, device_A, sizeof(float) * NUM_THREADS * NUM_BLOCKS, 
						cudaMemcpyDeviceToHost ); // copy(to, from, size, from_host_to_device direction)
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy to host vector A (error code is %s), doogdye.\n", cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
	
	for (i = 0; i < NUM_THREADS * NUM_BLOCKS; i++) {
		printf("host_A[%d] = %f\n", i, host_A[i]);
	}
	
	// Done...? (Also needs error-checking)
	err = cudaFree(device_A);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code is %s), goobdye.\n", cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
	free(host_A);

	return 0;
}