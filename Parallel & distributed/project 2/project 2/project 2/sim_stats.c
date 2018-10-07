/* sim_stats.c
 * Stephanie Taylor
 * Run a simulation and print the standard deviations of the CT6-crossing events.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"
#include "phase_support.h"
#include "my_timing.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: ./sim_stats <maxNumEvents> [<N>]\n");
    printf("  where\n");
    printf("       <maxNumEvents> is the maximum number of\n");
    printf("            events to report for each oscillator\n");
    printf("       <N> is the number of oscillators to simulate\n");
    printf("         (defaults to 100)\n");
    return 1;
  }
  int NmaxEvent = atoi( argv[1] );
  int Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  Params params;
  int i, j;
  float dt = 0.1; // 10 time steps per hour
  int numTimeSteps = 24*10*10+1; // 10 days
  params.periods = generateGradedPeriods( Nx ); 
  float *x0 = createInitialPhasesZeros( Nx );
  float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent );
  
  for (j = 0; j < NmaxEvent; j++)  {
    if (eventStds[j] == -1) {
        break;
    }
    printf( " %f", eventStds[j] );
  }
  printf( "\n" );

  free( eventStds );
  free( x0 );
  free( params.periods );
  return 0;
}
