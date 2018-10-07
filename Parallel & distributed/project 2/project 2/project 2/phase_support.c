#include "phase_support.h"
#include "my_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>


// Return a float array with Nx periods, starting at 24 h, and increasing by 0.1 h
float* generateGradedPeriods( int Nx ) {
  float* periods = (float *)malloc( Nx* sizeof(float) );
  int i;
  //printf("a\n");
  for (i=0; i < Nx; i++) {
    periods[i] = 24 + 0.1*i;
  }
  //printf("b\n");
  return periods;
}

// Return an array of Nx zeros.
float *createInitialPhasesZeros( int Nx ) {
  float *x0 = (float *)malloc( sizeof(float*) * Nx );
  int i;
  for (i=0; i < Nx; i++) {
    x0[i] = 0.0;
  }
  return x0;
}


float *runPhaseSimulation( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
	/*  as per Stephanie's reference guide */
   float *arr=(float *)malloc(sizeof(float)*Nx*Nt);
	int tidx;
	int cidx;
	for(tidx=0;tidx<Nt;tidx++){
		for(cidx=0;cidx<Nx;cidx++){
			float dx=1;
			if(tidx==0){
				arr[cidx]=x0[tidx];
			}
			else{
				arr[(tidx)*Nx+cidx]=arr[(tidx-1)*Nx+cidx]+dt*dx;
			}
		}
	}
    return arr;
}

int *runPhaseSimulationAndDetectEvents( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
    /* Credits to brandon for his help on this function  */
    int *arr=(int *)malloc(sizeof(int)*NmaxEvent*Nx);  // array of simulator events
    int *idxarr=(int *)malloc(sizeof(int)*Nx); // event count, stores which event the oscillator is on
    float *phasearr=(float *)malloc(sizeof(float)*Nt*Nx); // timeseries
    
    int tidx;
    int cidx;
    int ctr=0;
    float phase1,phase2;
    float cur,next,t;
    
    // run phase equation and search for possible events
    for(tidx=0;tidx<Nt;tidx++){ // time 
    	for(cidx=0;cidx<Nx;cidx++){  // loop through all oscillators
    		if(tidx==0){
    			phasearr[cidx]=x0[cidx];
    		}
    		else{
    			phasearr[tidx*Nx+cidx]=phasearr[(tidx-1)*Nx+cidx]+dt*1;
    			
    			t=paramStruct->periods[cidx];
    			phase1=phasearr[(tidx)*Nx+cidx];
    			phase2=phasearr[(tidx-1)*Nx+cidx];
    			cur=fmodf(phase1,t)*(24/t);
    			next=fmodf(phase2,t)*(24/t);
    			
    			if(cur>=6 && next<6){
    				if(idxarr[cidx]<NmaxEvent){
    					arr[cidx*NmaxEvent+idxarr[cidx]]=tidx;
    					idxarr[cidx]++;
    				}
    			}
    		}
    	}
    }
    free(phasearr);
    free(idxarr);
    return arr;
}

// Write this!
float *runPhaseSimulationFindEventStats( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
	
	float *StDevarr=(float *)malloc(sizeof(float)*Nx);
	// run phase simulation and get output array of timesteps
	int  *Tsteps=runPhaseSimulationAndDetectEvents(t0,dt,Nt,x0,Nx,paramStruct,NmaxEvent);
	
	float *events=(float *)malloc(sizeof(float)*Nx);
	
	int x,y;
	
	for(x=0;x<NmaxEvent;x++){
		for(y=0;y<Nx;y++){
			events[y]=Tsteps[y*NmaxEvent+x]*dt;
		}
		StDevarr[x]=std(events, Nx);
	}
	// avoid memory leaks
	free(Tsteps);
	free(events);
	return StDevarr;
	
}
