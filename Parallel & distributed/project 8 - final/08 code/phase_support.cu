#include <stdio.h>
#include <stdlib.h> 
#include "my_math.cuh"  // for the math functions
#include "sim_sizes.h" // for NX, NP, NE
#include "phase_support.cuh" // so we know about all the functions available to us
#include "math.h" // for fmod
#include "math_constants.h" // this is for PI

__constant__ __device__ float simulation_parameters_d[NP];

__host__ void setParams( float *params ) {
  cudaError_t err = cudaSuccess;
  err = cudaMemcpyToSymbol( simulation_parameters_d, params, sizeof(float)*NP );
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy to symbol area (error code %s)!\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize(); // This may not be necessary..
}


/* running with one block and NX threads. Each thread will be simulating a single oscillator  */
// run simulation for phase equation with a fixed initial configuration.
// Return the time each variable crosses CT6, heading up. 

__global__ void runPhaseSimulationAndDetectEvents( int Nt, int *events ) {
  // Declare shared vectors block
  __shared__ float xi[NX]; // phase at timestep i
  __shared__ float vi[NX]; // VIP output at timestep i
  float nextphase; // store current oscillator's phase for next step
  const float dt = 0.1; // constant timestep
  int eventCount = 0; // number of events seen
  int id = threadIdx.x; //one block of many threads

  float FRP = simulation_parameters_d[id];

  float vipStrength = simulation_parameters_d[NX]; // access the VIP strength
  xi[id] = 0.0; // initialize thread's phase

  // Initialize events to -1, only those entries associated with the idth oscillator
  int i;
  for (i = 0; i < NE; i++) {
      events[id*NE+i] = -1;
  }
	
  //make sure my other threads have finished their copying.
  __syncthreads();
  
    // Loop over all the time steps (starting at 1, ending when at Nt)
  int tidx;
  for (tidx = 1; tidx < Nt; tidx++) {
    // find out vip output (compute vi[id] from xi[id])
    float ctphase = fmodf(xi[id], FRP)*(24/FRP);
    if (ctphase >= 4 && ctphase <= 6) {
      vi[id] = vipStrength;
    }
    else {
      vi[id] = 0.0;
    }
    
    __syncthreads(); // wait for all vip outputs to be set

    // determine oscillator's nextphase, using its previous phase (xi[id])
    // phaseVelocity = 1+VIP*VRC
    float phaseVelocity = 1 + mean(vi, NX)*cosf(xi[id]*(M_PI*2/FRP)); 
    nextphase = xi[id] + dt*phaseVelocity;

    if (fmodf(xi[id], FRP)*(24/FRP) < 6 && fmodf(nextphase, FRP)*(24/FRP) > 6) {
      events[id*NE+eventCount] = tidx;
      eventCount++;
    }
    // set variables that will have us ready to use the newly calculated
    // values to get to the next step.
    // i.e the new value xip1 needs to go into xi[id]
    // update xi[id] with next phase
    xi[id] = nextphase;
  }
}
