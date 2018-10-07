/* sim_events.cu
 * This is the top-level program that gets the GPU to run the model and
 * return all the event times
 * Put parameters into Symbol memory (constant)
 * According to the documentation, the constant memory size is 64 KB (and that is
 * for the entire device, so we can't have a different set for each block). If we have 800 cells
 * (and just one SCN), then we _should_ have space for 20 floats for each cell.  This isn't enough to store an explicit network...
 * For each cell, we should have a period. And we should have one VIP-strength for all.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "my_math.cuh" // for functions that can be run either on the host or the device
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "sim_sizes.h" // For NX, NP, NE
#include "phase_support.cuh" // for the kernels
#include <sys/time.h> // for get_time_sec

// Return the time in seconds
// Note to Stephanie: it doesn't work to return a float - it must be
// a double.
double get_time_sec()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
} // end get_time_sec

int main(int argc, char* argv[]) {

  int i,j;
  cudaError_t err = cudaSuccess;

  if (argc < 2) {
    printf("Usage: sim_events <VIP>\n");
    printf("  where\n");
    printf("        <VIP> is the strength of the VIP output\n" );
    return 1;
  }

  // handle input
  float VIP = atof(argv[1]);
  int Nt = 24*10*NE+1;

  int *events_d;
  err = cudaMalloc((void **)&events_d, sizeof(int)*NX*NE);
  float *params_d;
  err = cudaMalloc((void **)&params_d, sizeof(float)*NP);

  int *events;
  events = (int *)malloc( sizeof(int)*NX*NE );
  float params[NP];
  for (i = 0; i < NX; i++) {
    params[i] = 24 + i*0.1;
  }
  params[NX] = VIP; // strength of VIP
  printf( "VIP = %f\n", VIP );
  
  double t1, t2;
  t1 = get_time_sec();
  
  // Get the input arguments to the device
  setParams( params );
  
  // Do all the work, here! ( Launch the CUDA Kernel)
  runPhaseSimulationAndDetectEvents<<<1,NX>>>(Nt,events_d);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch runPhaseEquationForDebug kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
  // Get the results from the device
  err = cudaMemcpy(events, events_d, sizeof(int)*NX*NE, cudaMemcpyDeviceToHost);

  t2 = get_time_sec();

  for (i = 0; i < NX; i++ ) {
    for (j = 0; j < NE; j++) {
        printf( "%d ", events[i*NE+j] );
    }
    printf( "\n" );
  }  
  printf( "It ran in %f seconds\n", t2-t1 );
    
//   printf( "params copied back from device\n" );
//   for (i=0; i<NX; i++) {
//     printf( "%f ", params[i] );
//   }
//   printf( "\n" );
  
  

  // clean up
  free(events);
//   free(params);
  cudaFree( events_d );
  cudaFree( params_d );

  return 0;
} //end main