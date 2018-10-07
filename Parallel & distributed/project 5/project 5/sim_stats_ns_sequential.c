/* sim_stats_ns.c
 * Stephanie Taylor
 * Run multiple simulations with different random period distributions and increasing VIP strengths. For each VIP strength, run a given number of simulations. For each simulation, print the standard deviation of the CT6-crossing of the final cycle of interest.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"
#include<pthread.h>
#include "phase_support.h"
#include "my_timing.h"

// global variables
Params params;
int numTrials;
float dt;
int numTimeSteps;
int numVIPs;
float *vipStrengths;
float *x0;
float *finalStds;



int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("Usage: ./sim_stats_ns <maxNumEvents> <N> <numTrials>\n");
    printf("  where\n");
    printf("       <maxNumEvents> is the maximum number of\n");
    printf("            events to report for each oscillator\n");
    printf("       <N> is the number of oscillators to simulate\n");
    printf("       <numTrials> the number of period distributions to simulate\n");
    return 1;
  }
  int NmaxEvent = atoi( argv[1] );
  int Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  params;
  params.periods = NULL;
  numTrials = atoi( argv[3] );
  int i, j;
  dt = 0.1;
  numTimeSteps = 24*10*(NmaxEvent*2);
  numVIPs = 21;
  vipStrengths = malloc( sizeof(float)*numVIPs );
  for (j = 0; j < numVIPs; j++) {
    	vipStrengths[j] = j*0.1;
  }
  x0 = createInitialPhasesZeros( Nx );
  finalStds = (float *)malloc( sizeof( float ) * numTrials * numVIPs );
  
  double start = get_time_sec();
  for (j = 0; j < numVIPs; j++) {
      params.vipStrength = vipStrengths[j];
      for (i = 0; i < numTrials; i++) {
          unsigned int seed = (i+1)*time(NULL);
  	  
      	  params.periods = generateGaussianPeriods( Nx, 24, 0.5,  &seed );
          float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent );
          finalStds[j*numTrials+i] = eventStds[NmaxEvent-1];
          free( eventStds );
          free( params.periods );
      }
  }
  double stop = get_time_sec();
  
  for (j=0; j < numVIPs; j++) {
    printf( "VIP strength %f: ", vipStrengths[j] );
    for (i = 0; i < numTrials; i++) {
      printf( " %f", finalStds[j*numTrials+i] );
    }
    printf( "\n" );
  }
  printf( "Ran in %f seconds\n", stop-start );

  free( x0 );
  return 0;
}
