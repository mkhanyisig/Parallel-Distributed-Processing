/* phase_io.c
 * Author: Stephanie Taylor
 * Has routines to write to and read from a binary file that stores
 * the output of a phase-only simulation. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "phase_io.h"


// Open the file, reading in all the data. Return a newly allocated
// Timeseries struct.
PhaseSimulationStruct *readPhaseSimulationFile(char *filename) {
  FILE *fp;

  if(filename != NULL && strlen(filename))
    fp = fopen(filename, "r");
  else
    return NULL;

  if (!fp)
    return NULL;

  PhaseSimulationStruct *ret = malloc(sizeof(PhaseSimulationStruct));
  fscanf( fp, "Nt=%d,Nx=%d\n", &(ret->Nt), 
         &(ret->Nx));
  ret->periods = (float *)malloc( sizeof( float ) * ret->Nx );
  ret->phases_RT = (float *)malloc( sizeof( float)*ret->Nx*ret->Nt );
  fread( ret->periods, sizeof(float), ret->Nx, fp );
  fread( ret->phases_RT, sizeof( float), ret->Nx*ret->Nt, fp );
  return ret;
} // end readPhaseSimulationFile

// Write the data to the file on disk (the open file is in *file). 
int writePhaseSimulationFile(char *filename, PhaseSimulationStruct *sim) {
  FILE *fp;

  if(filename != NULL && strlen(filename))
    fp = fopen(filename, "w");
  else
    return 0;
  
  if (!fp)
    return 0;

  fprintf(fp, "Nt=%d,Nx=%d\n",sim->Nt,sim->Nx);
  fwrite(sim->periods, sizeof(float), sim->Nx, fp);
  fwrite(sim->phases_RT, sizeof(float), sim->Nt*sim->Nx, fp);
  fclose(fp);
  
  return 1; // success!
} // end writePhaseSimulationFile


// dump all the phases to the Terminal
void dumpSimData(float *phases_RT, float *periods, int Nt, int Nx) {
  int tidx, xidx;
  for (tidx=0; tidx<Nt; tidx++) {
    for (xidx=0; xidx<Nx; xidx++ ) {
      printf("%f ",fmod(phases_RT[tidx*Nx+xidx], periods[xidx]));
    }
    printf( "\n" );
  }
}

void dumpPhaseSimulation( PhaseSimulationStruct *sim ) {
    dumpSimData( sim->phases_RT, sim->periods, sim->Nt, sim->Nx );
}





