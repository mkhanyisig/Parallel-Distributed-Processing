# Use Cuda 8.0's compiler
CC = /usr/local/cuda-8.0/bin/nvcc
# We need to link in the math library (Ok, not for this example, but we will eventually)
#LFLAGS = -lm -L/usr/local/cuda/lib64 -lcurand
LFLAGS = -lm -L/usr/local/cuda/lib64
# -m64 indicates this is a 64-bit architecture
# the sm_20 flag tells the compiler what our compute capability is,
CFLAGS = -m64 -gencode arch=compute_20,code=sm_20

# Target rules
all: build

build: hello

hello.o:	hello.cu
	$(CC) $(CFLAGS) -c -o $@ -c $<

vector_add.o:	vector_add.cu
	$(CC) $(CFLAGS) -c -o $@ -c $<
	
vector_dot_a.o:	vector_dot_a.cu
	$(CC) $(CFLAGS) -c -o $@ -c $<

vector_dot_b.o:	vector_dot_b.cu
	$(CC) $(CFLAGS) -c -o $@ -c $<

hello:	hello.o
	$(CC) $(CFLAGS) $(LFLAGS) -o hello hello.o

vector_add:	vector_add.o
	$(CC) $(CFLAGS) $(LFLAGS) -o vector_add vector_add.o

vector_dot_a:	vector_dot_a.o
	$(CC) $(CFLAGS) $(LFLAGS) -o vector_dot_a vector_dot_a.o

vector_dot_b:	vector_dot_b.o
	$(CC) $(CFLAGS) $(LFLAGS) -o vector_dot_b vector_dot_b.o

clean:
	rm -f  vector_add.o vector_add
	rm -f hello hello.o 
	rm -f  vector_dot_a.o vector_dot_a
	rm -f  vector_dot_b.o vector_dot_b

clobber: clean
