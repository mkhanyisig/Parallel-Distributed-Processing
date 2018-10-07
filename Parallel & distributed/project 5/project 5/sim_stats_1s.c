/* sim_stats_1s.c
 * Stephanie Taylor
 * Run multiple simulations with different random period distributions. For each simulation, print the standard deviation of the CT6-crossing of the final cycle of interest.
 *  Parallel version
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "phase_support.h"
#include "my_timing.h"

// global variables
float *finalStds;
int NmaxEvent;
int Nx;
Params params;
int numTrials;
float dt;
int numTimeSteps;
float *x0;


typedef struct thread {   
     Params p;
  	 int id;     
} ThreadData;

void *myThreadFun(void *thread){
    	
      ThreadData *my_data  = (ThreadData*)thread;
       int j;	 
	  int id = (int )my_data->id;

	  Params ps=(Params)my_data->p;
	  
	   unsigned int seed = (id+1)*time(NULL);
	  
      ps.periods = generateGaussianPeriods( Nx, 24, 0.5 , &seed);
      
      float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &ps,  NmaxEvent );
      for(j=0;j<NmaxEvent;j++){
      	printf("Event   #%d     stdev= %f\n",j,eventStds[j]);
      }
      finalStds[id] = eventStds[NmaxEvent-1];
      printf("final stats for trial %d:    %f\n",id,finalStds[id]);
      free( eventStds );
      free( ps.periods );
    
    return NULL;
}


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
  NmaxEvent = atoi( argv[1] );
  Nx = 100;
  if (argc > 2) {
    Nx = atoi( argv[2] );
  }
  params.vipStrength = atof( argv[3] );
  params.periods = NULL;
  numTrials = atoi( argv[4] );
  int i,j;
  dt = 0.1;
  numTimeSteps = 24*10*(NmaxEvent*2);
  x0 = createInitialPhasesZeros( Nx );
  finalStds = (float *)malloc( sizeof( float ) * numTrials );
  ThreadData thread[numTrials];
  pthread_t tid[numTrials];
  
  
  
  
  double start = get_time_sec();
  for (i = 0; i < numTrials; i++){
		thread[i].id=i;
		thread[i].p=params;
		printf("begin %d\n",i);
    	pthread_create(&tid[i], NULL, myThreadFun, (void *)&(thread[i]) );
    	printf("end %d\n",i);
 }
 
 	printf("joining threads\n");
    for (i=0;i<numTrials;i++){
    	pthread_join(tid[i],NULL);
	}
	  
  /* // old code
  for (i = 0; i < numTrials; i++) {
      params.periods = generateGaussianPeriods( Nx, 24, 0.5 );
      float *eventStds = runPhaseSimulationFindEventStats( 0.0, dt, numTimeSteps, x0, Nx,  &params,  NmaxEvent );
      for(j=0;j<NmaxEvent;j++){
      	printf("Event   #%d     stdev= %f\n",j,eventStds[j]);
      }
      finalStds[i] = eventStds[NmaxEvent-1];
      printf("final stats for trial %d:    %f\n",i,finalStds[i]);
      free( eventStds );
      free( params.periods );
  }
  */
  double stop = get_time_sec();
  
  for (i = 0; i < numTrials; i++) {
    printf( " %f", finalStds[i] );
  }
  printf( "\n" );
  printf( "Ran in %f seconds\n", stop-start );

  free( x0 );
  return 0;
}
