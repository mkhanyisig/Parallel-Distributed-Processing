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
	// o[Nx] holds strengths of the VIP relasess 
	float o[Nx],mod,VRC,v;
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
	 //printf("     ?######       %f    \n",paramStruct->vipStrength);
	v=mean(o,Nx);// add up the VIP inputs and then divide by the number of cells that send VIP to cells i
	for(i=0;i<Nx;i++){// VRC is a cosine curve
		VRC=cos(phases[i]*2*M_PI/paramStruct->periods[i]);
		next_phases[i]=1+v*VRC;
	}
}

float *runPhaseSimulation( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
	float* timeseries=(float*)malloc(sizeof(float)*(Nt+1)*Nx);   //(only difference here)
	
	int i,j,seriesI;
	float v, vrc,rhs;
	for(i=0;i<Nx;i++){
		timeseries[i]=x0[i];
	}
	for(i=0;i<Nt;i++){
		phaseEq(timeseries+i*Nx,timeseries+(i+1)*Nx,Nx,paramStruct);
		for(j=0,seriesI=(i+1)*Nx;j<Nx;j++,seriesI++){
			timeseries[seriesI]=timeseries[seriesI-Nx]+timeseries[seriesI]*dt;
		}
	}
	return timeseries;
}


float *runPhaseSimulation2( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
	/*  as per Stephanie's reference guide */
   float* arr=(float*)malloc(sizeof(float)*Nx*Nt);
	int tidx,cidx,i,seriesI;
	float v,VRC, rhs;
	for(i=0;i<Nx;i++){
		arr[i]=x0[i];
	}
	for(tidx=0;tidx<Nt;tidx++){
		phaseEq(arr+tidx*Nx,arr+(tidx+1)*Nx,Nx,paramStruct);
		for(cidx=0,seriesI=(i+1)*Nx;cidx<Nx;cidx++, seriesI++){
			arr[seriesI]=arr[seriesI-Nx]+arr[seriesI]*dt;
		}
	}
    return arr;
}

float *runPhaseSimulation3( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct )  {
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
    printf("here\n");
    float* timeseries=(float*)malloc(sizeof(float)*(Nt+1)*Nx);
    int* events=(int*)malloc(sizeof(int)*NmaxEvent*Nx);
    int eventCount[Nx];// counts the number of events for each phase
    int i, j, seriesI;
    float phase1,phase2;
    float cur,next,t,prev;
    printf("here\n");
    for(i=0;i<Nx;i++){
    	timeseries[i]=x0[i];
    	eventCount[i]=0;
    	//printf("**\n");
    	for(j=i*NmaxEvent;j<(i+1)*NmaxEvent;j++){
    		//printf("**\n");
    		events[j]=-1;
    	}
    	//printf("**\n");
    }
    printf("here\n");
    for(i=0;i<Nt;i++){
    	phaseEq(timeseries+i*Nx,timeseries+(i+1)*Nx,Nx,paramStruct);
    	for(j=0,seriesI=(i+1)*Nx;j<Nx;j++,seriesI++){
    		timeseries[seriesI]=timeseries[seriesI-Nx]+timeseries[seriesI]*dt;
    		 // stephanie's correction
    		if(eventCount[j]<NmaxEvent 
    		&& fmod(timeseries[seriesI-Nx],paramStruct->periods[j])*24/paramStruct->periods[j]  <= 6
    		&& fmod(timeseries[seriesI],paramStruct->periods[j])*24/paramStruct->periods[j] > 6){
    			events[NmaxEvent*j+eventCount[j]]=i+1;
    			eventCount[j]++;
    		}
    		/*
    		t=paramStruct->periods[j];
    		printf("here\n");
			phase1=timeseries[(i)*Nx+j];
			phase2=timeseries[(i-1)*Nx+j];
			printf("here2\n");
			cur=fmodf(phase1,t)*(24/t);
			prev=fmodf(phase2,t)*(24/t);
			printf("here3\n");
			//printf("   cur %f     next %f",cur,next);
			if(cur>=6 && prev<6){
				if( eventCount[j]<NmaxEvent){
					events[j*NmaxEvent+eventCount[j]]=i;
					eventCount[j]++;
				}
			}
			
			
			if(eventCount[j]<NmaxEvent 
    		&& fmod(timeseries[seriesI-Nx],paramStruct->periods[j])*4  <= paramStruct->periods[j]
    		&& fmod(timeseries[seriesI],paramStruct->periods[j])*4 > paramStruct->periods[j]){
    			events[NmaxEvent*j+eventCount[j]]=i+1;
    			eventCount[j]++;
    		}
    		*/
    	}
    }
    free(timeseries);
    printf("done\n");
    timeseries=NULL;
    return events;
}



float *runPhaseSimulationAndDetectEvents4( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
    /* help neeeded here, breaks if  I use other loop  */
	float* arr=(float*)malloc(sizeof(float)*NmaxEvent*Nx);  // array of simulator events
    int* idxarr=(int*)malloc(sizeof(int)*Nx); // event count, stores which event the oscillator is on
    float* phasearr=(float*)malloc(sizeof(float)*Nt*Nx); // timeseries
    
    int tidx;
    int cidx;
	int seriesI,i;
    int ctr=0;
    float phase1,phase2;
    float cur,next,t,prev;
    
    for(i=0;i<NmaxEvent*Nx;i++){
    	arr[i]=-1;
    }

	for(i=0;i<Nx;i++){
		phasearr[i]=x0[i];
		idxarr[i]=0;
	}

    // run phase equation and search for possible events
    for(tidx=1;tidx<Nt;tidx++){ // time 
    	phaseEq(phasearr+tidx*Nx,phasearr+(tidx+1)*Nx,Nx,paramStruct );   
		for(cidx=0,seriesI=(tidx+1)*Nx;cidx<Nx;cidx++,seriesI++){  // loop through all oscillators
    			phasearr[tidx*Nx+cidx]=phasearr[(tidx-1)*Nx+cidx]+dt*phasearr[tidx*Nx+cidx];
				//phasearr[seriesI]=phasearr[seriesI-Nx]+phasearr[seriesI]*dt;
				
    			t=paramStruct->periods[cidx];
    			phase1=phasearr[(tidx)*Nx+cidx];
    			phase2=phasearr[(tidx-1)*Nx+cidx];
    			cur=fmodf(phase1,t)*(24/t);
    			prev=fmodf(phase2,t)*(24/t);
    			//printf("   cur %f     next %f",cur,next);
    			if(cur>=6 && prev<6){
    				if(idxarr[cidx]<NmaxEvent){
    					arr[cidx*NmaxEvent+idxarr[cidx]]=tidx;
    					idxarr[cidx]++;
    				}
    			}
				
    		}
    	}
    printf("     ***********        %f    \n",paramStruct->vipStrength);
	printf("****** gets here\n");
    //free(phasearr);  // (breaks here as well)
    //free(idxarr);
    return arr;
}

float *runPhaseSimulationAndDetectEvents2( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
    /* Credits to brandon for his help on this function  */
    float *arr=(float *)malloc(sizeof(float)*NmaxEvent*Nx);  // array of simulator events
    int *idxarr=(int *)malloc(sizeof(int)*Nx); // event count, stores which event the oscillator is on
    float *phasearr=(float *)malloc(sizeof(float)*Nt*Nx); // timeseries
    
    int tidx;
    int cidx;
    int ctr=0;
    float phase1,phase2;
    float cur,next,t;
    int i;
    
    for(i=0;i<NmaxEvent*Nx;i++){
    	arr[i]=-1;
    }
    
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

float std_positive(float *array, int N, int interval){
	int actualN=N;
	float sum=0,average,stdev;
	int n,i;
	
	for(n=0,i=0;n<N;n++,i+=interval){
		if(array[i]>=0){ 
			sum+=array[i];
		}
		else{
			actualN--;
		}
	}
	
	if(actualN<=1){
		return 0;
	}
	average=sum/actualN;
	sum=0;
	for(n=0,i=0;n<N;n++,i+=interval){
		if(array[i]>=0){
			sum+=(array[i]-average)*(array[i]-average);
		}
	}
	stdev=sqrt(sum/(actualN-1));
	return stdev;
	

}


float *runPhaseSimulationFindEventStats( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
	
	
	float* timeseries=(float*)malloc(sizeof(float)*(Nt+1)*Nx);
	float* events=(float*)malloc(sizeof(float)*NmaxEvent*Nx);
	float* eventStats=(float*)malloc(sizeof(float)*NmaxEvent);
	int eventCount[Nx];
	int i,j,k, seriesI;
	float phase1,phase2;
    float cur,next,t,prev;
	float realTime;// records the real time that passes
	int days;
	float* tempTime;
	printf("begin\n");
	// initializations
	for(i=0;i<Nx;i++){
		timeseries[i]=x0[i];
		eventCount[i]=0;
		// initialize the events array with negative numbers	
		for(j=i*NmaxEvent; j<(i+1)*NmaxEvent;j++){
			events[j]=-1;
		}
	}
	printf("end\n");
	realTime=0;
	days=0;
	printf("begin 2\n");
	for(i=0;i<Nt;i++){
		phaseEq(timeseries+i*Nx,timeseries+(i+1)*Nx,Nx,paramStruct);
		//printf("passes\n");
		for(j=0,seriesI=(i+1)*Nx;j<Nx;j++,seriesI++){
			//printf("**\n");
			timeseries[seriesI]=timeseries[seriesI-Nx]+timeseries[seriesI]*dt;
			
			 // stephanie's correction
    		if(eventCount[j]<NmaxEvent 
    		&& fmod(timeseries[seriesI-Nx],paramStruct->periods[j])*24/paramStruct->periods[j]  <= 6
    		&& fmod(timeseries[seriesI],paramStruct->periods[j])*24/paramStruct->periods[j] > 6){
    			events[NmaxEvent*j+eventCount[j]]=i+1;
    			eventCount[j]++;
    		}
    		/*
    		// old
    		t=paramStruct->periods[j];
    		printf("1\n");
			phase1=timeseries[(i)*Nx+j];
			phase2=timeseries[(i+1)*Nx+j];
			printf("2\n");
			cur=fmodf(phase1,t)*(24/t);
			prev=fmodf(phase2,t)*(24/t);
			printf("3\n");
			//printf("   cur %f     next %f",cur,next);
			if(cur>=6 && prev<6){
				if( eventCount[j]<NmaxEvent){
					events[j*NmaxEvent+eventCount[j]]=i;
					eventCount[j]++;
				}
			}
			
			
			if(eventCount[j]<NmaxEvent 
    		&& fmod(timeseries[seriesI-Nx],paramStruct->periods[j])*4  <= paramStruct->periods[j]
    		&& fmod(timeseries[seriesI],paramStruct->periods[j])*4 > paramStruct->periods[j]){
    			events[NmaxEvent*j+eventCount[j]]=i+1;
    			eventCount[j]++;
    		}
    		*/
    		
		}	
	}
	printf("end2\n");
	// generate statistics
	for(i=0;i<NmaxEvent;i++){
		//printf("event   #%d   %f   \n",i,std_positive(events+i,Nx, NmaxEvent));
		eventStats[i]=std_positive(events+i,Nx, NmaxEvent);   
	}
	free(timeseries);
	free(events);
	timeseries=NULL;
	return eventStats;
}


float *runPhaseSimulationFindEventStats4( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent ) {
	/*  original   */
	
	//printf("   here \n");
	float *StDevarr=(float *)malloc(sizeof(float)*NmaxEvent);   // eventStats
	int  *Tsteps=runPhaseSimulationAndDetectEvents(t0,dt,Nt,x0,Nx,paramStruct,NmaxEvent);// run phase simulation and get output array of timesteps
	float *events=(float *)malloc(sizeof(float)*Nx);
	printf("   here \n");
	int x,y;
	int N = 0;
	
	for(x=0;x<NmaxEvent;x++){
		N = 0;
		
		for(y=0;y<Nx;y++){
			if (Tsteps[y*NmaxEvent+x] != -1) {
				events[N++]=Tsteps[y*NmaxEvent+x]*dt;
			}
			//printf("         *  %f       %d     %d\n",events[y],x,y);
			StDevarr[x]=std(events, N);
			//printf("event   %d    std  %f  \n",x,StDevarr[x]);
		}
		
	}
	// avoid memory leaks
	free(Tsteps);
	free(events);
	return StDevarr;
	
}

double randn (double mu, double sigma){
	/* In each iteration two normal random variables are generated. Therefore we can generate two random 
	variables in one iteration send one, and on the new call we will execute the algorithm and instead 
	we will return the second generated value from the previous call.   */
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1) {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do{
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}


// Adapted from the C++ code on Wikipedia
// https://en.wikipedia.org/wiki/Box–Muller_transform
double generateGaussian(double mu, double sigma)
{
	static const double epsilon = DBL_MIN;
	static const double two_pi = 2.0*3.14159265358979323846;

	double z1, u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

// Return a float array with Nx periods taken from a Gaussian distribution
// with the given mean and standard deviation.
float *generateGaussianPeriods( int Nx, float mean, float std_dev ){
	float* arr=(float*)malloc(sizeof(float)*Nx);
	int i;
	//printf(" Testing generated Gausian periods:\n",mean,std_dev);
	//printf("****\n");
	for(i=0;i<Nx;i++){
		arr[i]=(float)generateGaussian(mean,std_dev);
		//printf(" %f \n ",randn(mean,std_dev));
	}
	//printf("****\n");
	//printf("***********   DONE    ***********\n");
	return arr;
}

