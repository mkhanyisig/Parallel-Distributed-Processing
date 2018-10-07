/*
Mkhanyisi Gamedze
CS336 Parallel Processing
Project 6
*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include "my_timing.h"


/*
	 pass one fish at a time. The buffers between the workers can hold one fish
*/

#define NUM_WORKERS 5
#define NUM_FISH 200
#define BUFFER_SIZE 1

typedef struct{
	int id;
	double Tstart,Tend;
} Fish;

typedef struct{
	Fish** stock;
	int count;
} Buffer;

typedef struct{
	int id;
} Worker;

int fish_count=0;
Fish* fish;
int remaining_fish=NUM_FISH;
Buffer buffer[NUM_WORKERS-1];
pthread_mutex_t lock[NUM_WORKERS-1];
// initialise condition variables  
// ->condition variable always associated with a mutex, to avoid the race condition
pthread_cond_t nonFull[NUM_WORKERS-1];
pthread_cond_t nonEmpty[NUM_WORKERS-1];

void* worker_func(void* argz){
	Worker* info=(Worker*)argz;
	Fish* workee;
	int fish_count=0;
	printf( "fish %d\n", info->id );
	int finish=0;
	
	while(!finish){
		
		// consumer
		if(info->id>0){ // if not the first worker, consume
			
			// critical section with signalling
			// wait until previous buffer is full
			pthread_mutex_lock(&lock[info->id-1]);
			if(buffer[info->id-1].count<=0){
				// wait condition : empty buffer, wait till there is fish in it
				// automically unlock the mutex and wait for the condition variable to be signalled that buffer is nonEmpty
				pthread_cond_wait(&nonEmpty[info->id-1], &lock[info->id-1]); 
			}

			// fish in buffer, take it and do work
			buffer[info->id-1].count--;// take fish and do work
			workee=buffer[info->id-1].stock[buffer[info->id-1].count];
			pthread_cond_signal(&nonFull[info->id-1]);
			pthread_mutex_unlock(&lock[info->id-1]);
			
		}
		else{
 			if(remaining_fish>0){
 				remaining_fish--;
				workee=fish+fish_count;
				printf("fish  %d taken\n",remaining_fish);
				workee->Tstart=get_time_sec();
 			}
 			else{
 				printf("** No remaining fish: Done: wait for rest to finish**\n");
 			}
		}

		// do the work, prepare fish
		usleep((rand()%2+1)<<10);
		//printf("worker #%d   preparing fish  #%d\n",info->id,workee->id);
		
		// producer part
		if(info->id < NUM_WORKERS-1){// if not the last worker, produce
			// critical section
			pthread_mutex_lock(lock+info->id);
			if(buffer[info->id].count >= BUFFER_SIZE){// buffer is full, wait till it is not full to add fish onto it
				pthread_cond_wait(&nonFull[info->id], &lock[info->id]);
			}
			// add to buffer
			buffer[info->id].stock[buffer[info->id].count]=workee;// add fish object to buffer
			buffer[info->id].count++;
			pthread_cond_signal(&nonEmpty[info->id]);
			pthread_mutex_unlock(&lock[info->id]);
		}
		else {
			// if this is the last thread, then fish is entirely processed
			workee->Tend= get_time_sec();
			printf("Fish  %d is done \n",fish_count);
		}
		
		// break statement
		fish_count++;
		if(fish_count>NUM_FISH){
			break;
		}
	}
	return NULL;
}


int main(int argc, char *argv[]){
	srand(time(NULL));
	int i,j;
	double t1, t2;
	fish =malloc(sizeof(Fish)*NUM_FISH);
	for(i=0;i<NUM_FISH;i++){
		fish[i].id=i;
	}
	pthread_t workers[NUM_WORKERS];
	Worker worker_info[NUM_WORKERS];
	for(i=0;i<NUM_WORKERS-1;i++){
		buffer[i].stock=(Fish**)malloc(sizeof(Fish*)*BUFFER_SIZE);
		buffer[i].count=0;
		pthread_mutex_init(lock+i,NULL);
		pthread_cond_init(nonFull+i,NULL);
		pthread_cond_init(nonEmpty+i,NULL);
	}
	
	printf("creating threads\n");
	t1 = get_time_sec(); // start timer
	for(i=0;i<NUM_WORKERS;i++){
		worker_info[i].id=i;
		pthread_create(workers+i,NULL,worker_func,worker_info+i);
	}
	//finish=1;
	printf("**   joining threads\n");
	// wait for threads to execute
	for(i=0;i<NUM_WORKERS-1;i++){
		pthread_join(workers[i],NULL);
	} 
    t2 = get_time_sec();// end the timer 
	
	// free mutex locks and  condition variables
	
	printf("##   Freeing threads\n");
	for(i=0;i<NUM_WORKERS-1;i++){
		free(buffer[i].stock);
		pthread_mutex_destroy(&lock[i]);
		pthread_cond_destroy(&nonFull[i]);
		pthread_cond_destroy(&nonEmpty[i]);
	}
	// printing
	for(i=0;i<NUM_FISH;i++){
		// uncomment to see fish usage
		printf("Fish #%d used  times st %f    sp %f   :%fs \n",fish[i].id,fish[i].Tstart,fish[i].Tend,fish[i].Tend-fish[i].Tstart);
	}
	
	printf("\n**\nTotal run time for whole production process %fs\n**\n",t2-t1);
	
	free(fish);

}

