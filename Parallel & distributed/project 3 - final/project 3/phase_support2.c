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

void phaseEq(float* phases, float* next_phases, int Nx, Params* paramStruct){
	float o[Nx];  // holds strengths of the VIP relasess 
	float mod;
	float VRC;
	float v;
	int i;
	for(i=0;i<Nx;i++){
		//  use modular arithmetic to figure out how many hours into its current cycle it is
		// mod (φj , τj )24/τj
		mod=fmod(phases[i],paramStruct->periods[i])*24/paramStruct->periods[i];
		if(4<=mod && mod<=8){
			o[i]=paramStruct->vipStrength;
		}
		else{
			o[i]=0;
		}
	}
	v=mean(o,Nx);
	for(i=0;i<Nx;i++){// VRC is a cosine curve
		VRC=cos(phases[i]*2*M_PI/paramStruct->periods[i]);
		next_phases[i]=1+v*VRC;
	}
}



// Run a phase simulation from initial time t0, with a step of dt, for Nt time steps.
// Use x0 as the initial conditions for the Nx state variables. The paramStruct
// should contain the periods of all the oscillators.
// The return value is an array containing the phase of all Nx oscillators at each time step.
// All phases for the first time step are first, then all the phases of the second time step, etc.

/*
float *runPhaseSimulation( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
	//  as per Stephanie's reference guide 
	// remodified after getting help from Zhuofan
   float* arr=(float*)malloc(sizeof(float)*Nx*Nt);  // timeseries 
	int tidx;
	int cidx;
	int seriesI;
	float v, vrc, rhs;
	for(int i=0;i<Nx;i++){// Use x0 as the initial conditions for the Nx state variables. The paramStruct
		arr[i]=x0[i];
	}
	for(tidx=0;tidx<Nt;tidx++){
		phaseEq(arr+tidx*Nx,arr+(tidx+1)*Nx,Nx,paramStruct);
		for(cidx=0,seriesI=(tidx+1)*Nx;cidx<Nx;cidx++, seriesI++){
				arr[seriesI]=arr[seriesI-Nx]+arr[seriesI]*dt;
		}
	}
    return arr;
}
*/
  // old method
float *runPhaseSimulation( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
	//  as per Stephanie's reference guide 
   float *arr=(float *)malloc(sizeof(float)*Nx*Nt);  // timeseries 
	int tidx;
	int cidx;
	for(tidx=0;tidx<Nt;tidx++){
		for(cidx=0;cidx<Nx;cidx++){
			// phase velocity
			phaseEq(&arr[tidx*Nx],&arr[(tidx+1)*Nx],Nx,paramStruct);
			float dx=1; //
			if(tidx==0){// Use x0 as the initial conditions for the Nx state variables. The paramStruct
				arr[cidx]=x0[cidx];
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
