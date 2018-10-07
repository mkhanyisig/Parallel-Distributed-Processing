/* phase_io.h
 * Author: Stephanie Taylor
 * Has routines to write to and read from a binary file that stores
 * the output of a phase-only simulation. 
 */
#ifndef PHASE_IO_H

#define PHASE_IO_H

#include <stdio.h>

typedef struct {
  int  Nx;
  int  Nt;
  float *phases_RT; // the phase at each time step for each oscillator, in units of hours
  float *periods; // the period of each oscillator, in units of hours
} PhaseSimulationStruct;

// Open the file, reading in all the data. Return a newly allocated
// Timeseries struct.
PhaseSimulationStruct *readPhaseSimulationFile(char *filename);

// Write the data to the file on disk (the open file is in *file). 
// The data are in the "data" array. 
int writePhaseSimulationFile(char *filename, PhaseSimulationStruct *sim);

// dump all the phases to the Terminal
void dumpPhaseSimulation( PhaseSimulationStruct *sim );

#endif
