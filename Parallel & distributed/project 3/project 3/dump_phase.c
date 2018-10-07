#include <stdlib.h>
#include <stdio.h>
#include "phase_io.h"

/* Main */
int main(int argc, char *argv[]) {
  char *filename;

  if (argc < 2) {
    printf("Usage: dump_sim <filename>\n");
    printf("       <filename> should be a .phs file\n");
    return 1;
  }

  filename = argv[1];

  PhaseSimulationStruct *sim = readPhaseSimulationFile(filename);
  dumpPhaseSimulation( sim );

  return 0;
} // end main
