/* sim_stats_1s.c
 * Stephanie Taylor
 * Run multiple simulations with different random period distributions. For each simulation, print the standard deviation of the CT6-crossing of the final cycle of interest.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"
#include <time.h>
#include <sys/time.h>
#include "phase_support.h"
#include "my_timing.h"

int main(int argc, char* argv[]) {
  if (argc < 5) {
    printf("Usage: ./sim_stats_1s <maxNumEvents> <N> <vipStrength> <numTrials>\n");
    printf("  where\n");
    printf("       <maxNumEvents> is the maximum number of\n");
    printf("            events to report for each oscillator\n");
    printf("       <N> is the number of oscillators to simulate\n");
    printf("       <vipStrength> is the floating point strength of VIP\n");
    printf("       <numTrials> the number of period distributions to simulate\n");
    return 1;
  }
  int NmaxEvent = atoi( argv[1] );
  int Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  Params params;
  params.vipStrength = atof( argv[3] );
  params.periods = NULL;
  int numTrials = atoi( argv[4] );
  int i,j;
  float dt = 0.1;
  int numTimeSteps = 24*10*(NmaxEvent*2);
  float *x0 = createInitialPhasesZeros( Nx );
  float *finalStds = (float *)malloc( sizeof( float ) * numTrials );
  unsigned int seed = (id+1)*time(NULL); // whats id?  -> thread ID
  
  double start = get_time_sec();
  for (i = 0; i < numTrials; i++) {
      params.periods = generateGaussianPeriods( Nx, 24, 0.5 ); // pass &seed
      float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent );
      for(j=0;j<NmaxEvent;j++){
      	printf("Event   #%d     stdev= %f\n",j,eventStds[j]);
      }
      finalStds[i] = eventStds[NmaxEvent-1];
      printf("final stats for trial %d:    %f\n",i,finalStds[i]);
      free( eventStds );
      free( params.periods );
  }
  double stop = get_time_sec();
  
  for (i = 0; i < numTrials; i++) {
    printf( " %f", finalStds[i] );
  }
  printf( "\n" );
  printf( "Ran in %f seconds\n", stop-start );

  free( x0 );
  return 0;
}
