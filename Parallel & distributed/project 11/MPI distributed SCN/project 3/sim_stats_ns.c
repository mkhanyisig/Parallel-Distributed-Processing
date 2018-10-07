/* sim_stats_ns.c
 * Stephanie Taylor
 * project 11 Mkhanyisi Gamedze
 * Run multiple simulations with different random period distributions and increasing VIP strengths. For each VIP strength, run a given number of simulations. For each simulation, print the standard deviation of the CT6-crossing of the final cycle of interest.
 */
#include <mpi.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"
#include "phase_support.h"
#include "my_timing.h"

int main(int argc, char* argv[]) {
  /* 
  // old P3 code
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
  Params params;
  params.periods = NULL;
  int numTrials = atoi( argv[3] );
  int i, j;
  float dt = 0.1;
  int numTimeSteps = 24*10*(NmaxEvent*2);
  int numVIPs = 21;
  float *vipStrengths = malloc( sizeof(float)*numVIPs );
  for (j = 0; j < numVIPs; j++) {
    vipStrengths[j] = j*0.1;
  }
  float *x0 = createInitialPhasesZeros( Nx );
  
  float *finalStds = (float *)malloc( sizeof( float ) * numTrials * numVIPs );
  
  // start simulation
  double start = get_time_sec();
  for (j = 0; j < numVIPs; j++) {
      params.vipStrength = vipStrengths[j];
      for (i = 0; i < numTrials; i++) {
          params.periods = generateGaussianPeriods( Nx, 24, 0.25 );
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
  */
  
  // new MPI code
  srand(time(NULL));
  
  int num_procs, myrank;
  MPI_Init(&argc, &argv); // start MPI distributed processes
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );

  if (argc < 4) {
    if (myrank == 0) { // only print once in rank 0
      printf("Usage: sim_stats_ns <maxNumEvents> <N> <numTrials>\n");
      printf("  where\n");
      printf("       <maxNumEvents> is the maximum number of\n");
      printf("            events to report for each oscillator\n");
      printf("       <N> is the number of oscillators to simulate\n");
      printf("       <numTrials> the number of period distributions to simulate\n");
    }
    MPI_Finalize(); // end MPI error made
    return 0;
  }
  // assign variables
  int NmaxEvent = atoi( argv[1] );
  int Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  int numTrials = atoi( argv[3] );
  
  
  Params params;
  params.periods = NULL;
  int i, j;
  float dt = 0.1;
  int numTimeSteps = 24*10*(NmaxEvent*2);
  int numVIPs = 21;
  float *vipStrengths = malloc( sizeof(float)*numVIPs );
  for (j = 0; j < numVIPs; j++) {
    vipStrengths[j] = j*0.1;
  }
  float *x0 = createInitialPhasesZeros( Nx );
  float *finalStds = (float *)malloc( sizeof( float ) * numTrials * numVIPs );
  
  
  // assign trials to process
  int start, end;
  int size = numTrials*numVIPs/num_procs;
  start = myrank*size;
  end = start + size - 1;
  if (myrank == 15) {
     end = numTrials*numVIPs-1;
  }
  float *processStds = malloc(sizeof(float)*(end-start+1));
  double start_sim = MPI_Wtime();
  // run phase simulation
  for (i = start; i <= end; i++) {
    params.periods = generateGaussianPeriods( Nx, 24, 0.5 );
    // run simulation for certain VIP strength
    params.vipStrength = vipStrengths[i/numTrials];
    float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent );
    processStds[i-start] = eventStds[NmaxEvent-1];
    free( eventStds ); // no longer needed
  }
  double end_sim = MPI_Wtime();
  printf("simulation  in proc_id %d took %f\n",myrank,end_sim-start_sim);
  // gather simulation stats
  int sentcount = end-start+1;
  int *sentcounts = malloc(sizeof(int)*num_procs);
  MPI_Gather(&sentcount, 1, MPI_INT, sentcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int *displs = malloc(sizeof(int)*num_procs);
  MPI_Gather(&start, 1, MPI_INT, displs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(processStds, sentcount, MPI_FLOAT, finalStds, sentcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  if(myrank==0){// print out final Stds
	  for (j=0; j < numVIPs; j++) {
		printf( "VIP strength %f: ", vipStrengths[j] );
		for (i = 0; i < numTrials; i++) {
		  printf( " %f\n", finalStds[j*numTrials+i] );
		}  
	  }
  }
  MPI_Finalize(); // end MPI
}
