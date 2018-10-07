/* sim_slow.c
 * Stephanie Taylor
 * Run a simulation and save the value of every phase at every timestep.
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
    printf("Usage: ./sim_slow  [ []] \n");
    printf("  where\n");
    printf("        id's the file which stores the output\n");
    printf("            Use the .phs extension in this filename\n");
    printf("        is the number of oscillators to simulate\n");
    printf("         (defaults to 100)\n");
    printf("        is the floating point strength of VIP\n");
    printf("            (default value is 0.0)\n");
    return 1;
  }
  char *filename = argv[1];
  int Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  Params params;
  if (argc > 3) {
    params.vipStrength = atof( argv[3] );
    printf( "setting vip strength %f\n", params.vipStrength );
  }
  else {
    params.vipStrength = 0.0;
  }
  int i;
  float dt = 0.1;
  int numTimeSteps = 24*10*10+1;
  //printf("gets here\n");
  params.periods = generateGradedPeriods( Nx );
  //printf("here too\n");
  float *x0 = createInitialPhasesZeros( Nx );
  //printf("passes this\n");
  float *ret = runPhaseSimulation( 0.0, dt, numTimeSteps, x0, Nx, &params ); 
  //printf("here as well\n");
  printf( "ret %f %f\n", ret[0], ret[1] );
  printf("gets here\n");
  PhaseSimulationStruct sim;
  sim.Nt  = numTimeSteps;
  sim.Nx = Nx;
  sim.phases_RT = ret;
  sim.periods = params.periods;
  printf("here as well\n");
  writePhaseSimulationFile( filename, &sim );
   printf("here \n");
  free( ret );
  free( x0 );
  free( params.periods );
  return 0;
}
