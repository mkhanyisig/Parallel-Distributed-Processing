/* phase_support.cuh:
 */

#ifndef PHASE_SUPPORT_H

#define PHASE_SUPPORT_H


// prototypes
__host__ void setParams( float *params );

__global__ void runPhaseSimulationAndDetectEvents( int Nt, int *events );



#endif