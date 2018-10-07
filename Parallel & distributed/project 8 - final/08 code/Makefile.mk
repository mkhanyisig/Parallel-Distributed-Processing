# This makefile is setup to be run on bombur.cs.colby.edu-n1, n2, n3, or n4.
# If you need to run it on a different computer, then use a different path to
# the compiler and a different compute capability.

# Use Cuda's compiler
CC = /usr/local/cuda-8.0/bin/nvcc
# We need to link in the math library
LFLAGS = -lm -L/usr/local/cuda/lib64
# If we were to use random numbers generated on the device, then we would need to link in the Curand library.
# LFLAGS = -lm -L/usr/local/cuda/lib64 -lcurand
# the sm_20 flag tells the compiler what our compute capability is,
# so it knows it can use things like printf in phase_support.
# -m64 indicates this is a 64-bit architecture
CFLAGS = -m64 -gencode arch=compute_20,code=sm_20 
OBJS = phase_support.o my_math.o
EXECS = sim_events 

phase_support.o :	phase_support.cu sim_sizes.h
	$(CC) -x cu $(CFLAGS) -dc -o phase_support.o phase_support.cu

my_math.o :	my_math.cu sim_sizes.h
	$(CC) -x cu $(CFLAGS) -dc -o my_math.o my_math.cu

sim_events:	sim_events.cu sim_sizes.h $(OBJS)  
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS)  -o sim_events sim_events.cu

all :	$(EXECS)

clean :
	rm -rf *~ *.o $(EXECS)

# Inference rules
%.o : %.cu 
	$(CC) -x cu $(CFLAGS) -dc -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $^