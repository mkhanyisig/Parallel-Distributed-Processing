CC = gcc
LFLAGS = -lm -lpthread
CFLAGS =
OBJS = phase_support.o my_timing.o my_math.o phase_io.o

EXECS = sim_slow dump_phase sim_events sim_stats disp_phase

sim_slow :	sim_slow.c $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o sim_slow sim_slow.c

sim_events :	sim_events.c $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o sim_events sim_events.c

sim_stats :	sim_stats.c $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o sim_stats sim_stats.c

sim_stats_1s :	sim_stats_1s.c $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o sim_stats_1s sim_stats_1s.c

sim_stats_ns :	sim_stats_ns.c $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o sim_stats_ns sim_stats_ns.c

dump_phase:   dump_phase.c phase_io.o
	$(CC) $(CFLAGS) $(LFLAGS) phase_io.o -o dump_phase dump_phase.c

disp_phase: disp_phase.o phase_io.o my_timing.o
	gcc -o $@ $^ -L/usr/X11R6/lib -lX11 -lm

clean :
	rm -rf *~ *.o $(EXECS)
	
all: $(EXECS)

# Inference rules
%.o : %.c
	$(CC) $(CFLAGS)-c -o $@ $^
 