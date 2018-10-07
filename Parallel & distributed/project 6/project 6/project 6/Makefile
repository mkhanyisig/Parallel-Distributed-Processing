# makefile
CC = gcc
LFLAGS = -lm  -lpthread
CFLAGS =
OBJS =

fish_production_single : fish_production_single.c my_timing.o
	$(CC)  $(CFLAGS) my_timing.o -o fish_production_single fish_production_single.c
	
fish_production_batch : fish_production_batch.c my_timing.o
	$(CC)  $(CFLAGS) my_timing.o -o fish_production_batch fish_production_batch.c

fish_production_multiple : fish_production_multiple.c my_timing.o
	$(CC)  $(CFLAGS) my_timing.o -o fish_production_multiple fish_production_multiple.c

clean :
	rm -rf *~ *.o fish_production_single fish_production_multiple  fish_production_batch
	
all: $(EXECS)

# Inference rules
%.o : %.c
	$(CC) $(CFLAGS)-c -o $@ $^
