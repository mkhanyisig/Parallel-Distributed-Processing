/* sim_events.c
 * Stephanie Taylor
 * Run a simulation and save the indices of the time steps of the events.
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
    printf("Usage: ./sim_events <maxNumEvents> [N]\n");
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
  params.vipStrength = 1.5;
  int i, j;
  float dt = 0.1;
  int numTimeSteps = 24*10*10+1;
  params.periods = generateGaussianPeriods( Nx, 24.0, 0.5 );
  float *x0 = createInitialPhasesZeros( Nx );
  double start = get_time_sec(); // record time before simulation
  int *events = runPhaseSimulationAndDetectEvents( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent ); // record time after simulation
  double stop = get_time_sec();
  
  for (i = 0; i < Nx; i++) {
    printf( "Oscillator %d:", i );
    for (j = 0; j < NmaxEvent; j++)  {
        if (events[i*NmaxEvent+j] == -1) {
            break;
        }
        printf( " %d", events[i*NmaxEvent+j] );
    }
    printf( "\n" );
  }

  printf( "Ran in %f seconds\n", stop-start ); // report difference (unit is seconds)
  free( events );
  free( x0 );
  free( params.periods );
  return 0;
}
