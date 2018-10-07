/*
	Mkhanyisi Gamedze
	CS336 : Parallel and Distributed systems
	Project 9
	June 2018
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

int randInt(int min, int max){// return random integer between min & max    
    int num = (rand() %(max - min + 1)) + min;
    //printf("num=%d\n",num);
    return num;
}

double get_time_sec(){// get current time
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    return t.tv_sec + t.tv_usec*1e-6;
}

int counter(int * arr, int size){// given array, count the number of 3s
    int i;
    int count=0;
    double start=get_time_sec();
    for(i=0;i<size;i++){// count 3s
        if(arr[i]==3)
            count++;
    }
    double end=get_time_sec();
    printf("sequential count took %fs\n",end-start);
    return count;
}


int main(int argc, char ** argv){
    if(argc<2){
        printf("count3s <NUM_SIZE>\n");
        return -1;
    }
    int size=atoi(argv[1]);
    
    int i,sentcount, recvcount,numtasks, rank,source;
    
    
    double mpi_start=get_time_sec();
    MPI_Init(&argc,&argv); // intiate MPI
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);// find total num processors
    
    // allocate memory for variables
    double startup_start=get_time_sec();
    int* sentbuf=(int*)malloc(sizeof(int)*size);
    int* recvbuf=(int*)malloc(sizeof(int)*(size/numtasks));
    int* globalcount=(int*)malloc(sizeof(int)*numtasks);
    int* count=(int*)malloc(sizeof(int));
    double startup_end=get_time_sec();
    printf("** startup time  %f\n",startup_end-startup_start);
    
    //  Allocate an array of randomly generated integers between 0 and 9 in the root process
    double time;
    if(rank==0){// root process
    	printf("starting problem details:  size=%d, numtasks=%d\n",size,numtasks);
    	//populate array
    	double init_time=get_time_sec();
        for(i=0;i<size;i++){
            *(sentbuf+i)=randInt(0,10);
        }
        double end_time=get_time_sec();
        printf("** filling array, took %fs\n",end_time-init_time);
        time=MPI_Wtime();	
    }
    // scatter it to the rest of the processors
    // root process sending data to all processes in the communicator
    double scatter_start=get_time_sec();
    MPI_Scatter(sentbuf,(size/numtasks),MPI_INT,recvbuf,(size/numtasks),MPI_INT,0, MPI_COMM_WORLD);
    double scatter_end=get_time_sec();
    //printf("scattering took   %f\n",scatter_end-scatter_start);
    
    *count=0;
    double rank_start=MPI_Wtime();
    for(i=0;i<(size/numtasks);i++){
        //printf("tid=%d,arr[%d]=%d\n",rank,i,recvbuf[i]);
        if(recvbuf[i]==3){// if received value is 3, increase count
            (*count)++;
        }
    }
    double rank_stop=MPI_Wtime();
    printf("						rank=%d,time=%fs\n", rank, rank_stop-rank_start);
    
    // gather counts
    double gather_start=get_time_sec();
    MPI_Gather(count,1,MPI_INT,globalcount,1,MPI_INT,0,MPI_COMM_WORLD);
    double gather_end=get_time_sec();
    //printf("gathering took   %f\n",gather_end-gather_start);
    
    double finalcounttime;
    if(rank==0){// back to root, count 3s
        *count=0;
        finalcounttime=MPI_Wtime();
        for(i=0;i<numtasks;i++){
        	//printf("gcount  %d\n",globalcount[i]);
            *count+=globalcount[i]; 
        }
        finalcounttime=MPI_Wtime()-finalcounttime;
        printf("** final count time on root process (rank0), took %fs\n",finalcounttime);

        printf("						count=%d,  time=%fs\n",*count,MPI_Wtime()-time);
        printf("actual count should be=%d\n",counter(sentbuf,size));
    }
    
    double cleanup_start=get_time_sec();
    free(sentbuf);
    free(recvbuf);
    free(count);
    free(globalcount);
    double cleanup_stop=get_time_sec();
    printf("** clean-up time  %f\n",cleanup_stop-cleanup_start);
    
    double mpi_stop=get_time_sec();
    printf("** MPI total run time  %f\n",mpi_stop-mpi_start);
    
    MPI_Finalize(); // end MPI  
    
}








