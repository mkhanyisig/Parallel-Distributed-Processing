/*
	Mkhanyisi Gamedze
	CS336 : Parallel and Distributed systems
	Project 10
	June 2018
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define MAX_VALUE 64000

//macro to calculate the bin x goes into
#define BIN(X,BIN_RANGE) (X/BIN_RANGE)



int randInt(int min, int max){// return random integer between min & max    
    int num = (rand() %(max - min + 1)) + min;
    //printf("num=%d\n",num);
    return num;
}

int square(int a){
	printf("squaring a=%d\n",a);
	return a*a;
}

double get_time_sec(){// get current time
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t, &tz);
    return t.tv_sec + t.tv_usec*1e-6;
}

// Comparison function used by qsort
int compare_ints(const void* arg1, const void* arg2){
  int a1 = *(int *)arg1;
  int a2 = *(int *)arg2;
  if (a1 < a2 ) return -1;
  else if (a1 == a2 ) return 0;
  else return 1;
}

// Sort the array in place
void qsort_ints(int *array, int array_len){
	qsort(array, (size_t)array_len, sizeof(int), compare_ints);
}

int sorted(int*array,int size){// check if the whole array is properly sorted
    int i=0;
    for(i=0;i<size-1;i++){
        if (array[i]>array[i+1]){
            return 0;
        }
    }
    return 1;
}

int main(int argc, char ** argv){
    
    if(argc<2){
        printf("bucket_sort <NUM_SIZE>\n");
        return -1;
    } 
    MPI_Init(&argc,&argv);// intiate MPI
    int size=atoi(argv[1]);
    int i, sendcount, recvcount, numtasks, rank, source;
    double time;
    int previous=0;
    struct timeval tv;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // find rank ID
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); // find total num processors
    const int bs=MAX_VALUE/numtasks; //bin size/range
    
    // fields
    int *values=(int*)malloc(sizeof(int)*size);
    int **bins=(int**)malloc(sizeof(int*)*numtasks); //numtasks= number of bins
    int *count=(int*)malloc(sizeof(int)*numtasks);
    int* displs=(int*)malloc(sizeof(int)*numtasks);
    int* recvbuf; //thread actual data 
    int * localcount=(int*)malloc(sizeof(int));
    
    
    if(rank==0){
    	printf("starting problem size=%d, numtasks=%d\n",size,numtasks);
        bzero(count,numtasks*sizeof(int));
        
        double pop_start=MPI_Wtime();
        for(i=0;i<size;i++){// populate values array
        	gettimeofday(&tv,NULL);
            srand(tv.tv_usec);
            int val, square;
            val=rand() % (int)sqrt(MAX_VALUE);
            square=val*val;
            values[i]=square; // generate value
            //printf("s %d\n",square);
            count[BIN(values[i],bs)]++; //increase bin count
        }
        double pop_stop=MPI_Wtime();
        printf("initiating values : %fs\n",pop_stop-pop_start);
        
        double div_start=MPI_Wtime();
        for(i=0;i<numtasks;i++){
            // add extra int to keep track of where to add
            // allocate just as much memory needed for each bin
            *(bins+i)=(int*)malloc(sizeof(int)*(*(count+i)+1)); 
            *(*(bins+i))=1;
        }
        // put elements into bins.
        //each bin's range: MAX_VALUE/numtasks
        for(i=0;i<size;i++){
        	bins[BIN(values[i],bs)][bins[BIN(values[i],bs)][0]]=values[i];
        	bins[BIN(values[i],bs)][0]++;
        }
        double div_stop=MPI_Wtime();
        printf("bin division : %fs\n",div_stop-div_start);
        
        bzero(values,size*sizeof(int)); // place size zero-valued bytes in the area pointed to by values
        double repop_start=MPI_Wtime();
        for(i=0;i<numtasks;i++){
        	displs[i]=previous;
        	//copy data to values
            memcpy(values+previous,(*(bins+i))+1,count[i]*sizeof(int));
            previous+=count[i];
        }
        double repop_stop=MPI_Wtime();
        printf("re-updating values : %fs\n",repop_stop-repop_start);
        // clean up bins
        for(i=0;i<numtasks;i++){
            free(*(bins+i));
        }
        free(bins);
        time=MPI_Wtime(); // bucket sort start
    }
    
   	// sending data to all processes in a communicator. sends chunks of our array to different processes
    MPI_Scatter(count,1,MPI_INT,localcount,1,MPI_INT,0,MPI_COMM_WORLD);
    //printf("gets here, rank %d\n",rank);
      
    recvbuf=(int*)malloc(sizeof(int)*(*localcount));
    // Scatters a buffer in parts to all processes in a communicator
    MPI_Scatterv(values,count,displs,MPI_INT,recvbuf, *localcount, MPI_INT,0,MPI_COMM_WORLD);
     
    // qsort each recvbuf bin : bucket sort algorithm
    qsort_ints(recvbuf,*localcount);
    
    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(recvbuf,*localcount,MPI_INT, values, count,displs,MPI_INT,0,MPI_COMM_WORLD);	
    
    if(rank==0){// root node has sorted array, check if that is true
    	// reached end, stop bucket sort
        printf("actual sorting : %fs\n",MPI_Wtime()-time);
        // quality check 
        if(sorted(values,size)){
            printf("correctly sorted: Success!!\n");
        }
        else{
        	printf(" something wrong, not properly sorted \n");
        }
        //clean up:
        free(displs);
        free(values);
        free(count);
    } 
    free(recvbuf);
    free(localcount);
    
    //printf("hello\n");
    MPI_Finalize(); // end MPI  
    return 0;
}
