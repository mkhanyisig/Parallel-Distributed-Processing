__global__ void runPhaseSimulationAndDetectEvents2( int Nt, int *events ) {
  // Declare shared vectors for 
  __shared__ float xi[NX]; // phase at timestep i
  __shared__ float vi[NX]; // VIP output at timestep i
  float xip1; // store this oscillator's phase for next step
  const float dt = 0.1; // Let's just make this constant
  int eventCount = 0; // This is for keeping track of how many events we have seen
  float v;// mean VIP
  float phase1,phase2;
  // Let's just use our within-block thread id to figure out which variable is "ours"
  // (I.e. assume the number of threads is Nx)
  int id = threadIdx.x;
  float FRP = simulation_parameters_d[id]; // access this thread's period
  xi[id] = 0.0; // initialize this thread's phase

  // Initialize events to -1. This thread should
  // initialize only those entries associated with the idth oscillator  
  // CODE HERE
  events[id]=-1;

  // Before we can start the simulation, I need to make sure my compatriots have
  // finished their copying.
  __syncthreads();
  
  // Loop over all the time steps (starting at 1, ending when at Nt)
  
  int tidx;
  for (tidx = 1; tidx < Nt; tidx++) {
		// figure out everyone's vip output (set vi[id], using xi[id])
		// CODE HERE
		vi[id]=xi[id];
		v=0;// VIP average

		__syncthreads(); // necessary, but why?

		// determine this oscillator's new phase xip1, using its previous phase (xi[id]). It depends on all VIP output (vi).
		// 
		// CODE HERE (set xip1, using xi[id] and vi)

		// check to see if we just encountered an event. If so, update the 
		// appropriate spot in events. Also, update eventCount.
		// CODE HERE

		// set variables that will have us ready to use the newly calculated
		// values to get to the next step.
		// i.e the new value xip1 needs to go into xi[id]
		xi[id] = xip1;
  }
}
