/* phase_support.h
 * Put your name here.
 */

#ifndef PHASE_SUPPORT_H

#define PHASE_SUPPORT_H

// Make a struct that will hold the period values.
// At the moment, this doesn't need to be a struct, but Stephanie
// is designing it to be one so that students can add additional
// parameters for extensions or later.
typedef struct{
  float *periods;
  float vipStrength;
} Params;

// Return a float array with Nx periods, starting at 24 h, and increasing by 0.1 h
float *generateGradedPeriods( int Nx );

// Return an array of Nx zeros.
float *createInitialPhasesZeros( int Nx );

// Run a phase simulation from initial time t0, with a step of dt, for Nt time steps.
// Use x0 as the initial conditions for the Nx state variables. The paramStruct
// should contain the periods of all the oscillators.
// The return value is an array containing the phase of all Nx oscillators at each time step.
// All phases for the first time step are first, then all the phases of the second time step, etc.
float *runPhaseSimulation( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct );

// Run a phase simulation from initial time t0, with a step of dt, for Nt time steps.
// Use x0 as the initial conditions for the Nx state variables. The paramStruct
// should contain the periods of all the oscillators.
// This version also detects the time steps at which each oscillator's phase passes through CT6 
// "on the way up". It reports a maximum of NmaxEvents event timestep indices for each oscillator. A value
// of -1 indicates an empty spot (no event detected).
// The events are grouped by oscillator.
float *runPhaseSimulationAndDetectEvents2( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent );// Run a phase simulation from initial time t0, with a step of dt, for Nt time steps.
int *runPhaseSimulationAndDetectEvents( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent );
// Use x0 as the initial conditions for the Nx state variables. The paramStruct
// should contain the periods of all the oscillators.
// This version also detects the time steps at which each oscillator's phase passes through CT6 
// "on the way up". It then calculates the standard deviation of the Nx event times for each cycle. It returns an array of NmaxEvents floats. (Note that the unit of the standard deviation will be hours).
float *runPhaseSimulationFindEventStats( float t0, float dt, int Nt, float *x0, int Nx, Params *paramStruct, int NmaxEvent );

// Return a float array with Nx periods taken from a Gaussian distribution
// with the given mean and standard deviation.
float *generateGaussianPeriods( int Nx, float mean, float std_dev );

#endif
