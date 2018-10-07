void* worker_func(void* argz){
	Worker* info=(Worker*)argz;
	Fish* workee[BUFFER_SIZE];
	int fish_count=0;
	printf( "fish %d\n", info->id );
	int finish=0;
	
	while(!finish){
		
		// consumer
		if(info->id>0){ // if not the first worker, consume
			printf("consumer : not first worker\n");
			
			// critical section with signalling
			// wait until previous buffer is full
			pthread_mutex_lock(&lock[info->id-1]);
			if(buffer[info->id-1].count<=0){
				// wait condition : empty buffer, wait till there is fish in it
				// automically unlock the mutex and wait for the condition variable to be signalled that buffer is nonEmpty
				pthread_cond_wait(&nonEmpty[info->id-1], &lock[info->id-1]); 
			}

			// fish in buffer, take it and do work
			printf("** buffer count  : %d\n",buffer[info->id-1].count);
			buffer[info->id-1].count-=1;// take a single fish from previous buffer and work on it
			printf("passes\n");
			workee[buffer[info->id].count-1]=buffer[info->id-1].stock[buffer[info->id-1].count+1];
			printf("gets here\n");
			pthread_cond_signal(&nonFull[info->id-1]);
			pthread_mutex_unlock(&lock[info->id-1]);
			printf("finish\n");	
		}
		
		else{
			printf("consumer : first worker\n");
 			if(remaining_fish>0){// take fish and do work
 				remaining_fish-=1;
 				buffer[info->id-1].count-=1;// reduce fish count on the buffer
				workee[buffer[info->id].count-1]=fish+fish_count+1;
				printf("fish  %d taken\n",remaining_fish+buffer[info->id-1].count);
				workee[buffer[info->id].count-1]->Tstart=get_time_sec();
				printf("*passes\n")	;
 			}
 			else{
 				printf("** No remaining fish: Done: wait for rest to finish**\n");
 			}
		}

		// do the work, prepare fish
		usleep(((rand()%2+1)*10)*BUFFER_SIZE);
		//printf("worker #%d   preparing fish  #%d\n",info->id,workee->id);
		
		// producer part
		if(info->id < NUM_WORKERS-1){// if not the last worker, produce and put fish onto next buffer
			//printf("producer: not last worker\n");
			// critical section
			pthread_mutex_lock(lock+info->id);
			if(buffer[info->id].count >= BUFFER_SIZE){// buffer is full, wait till it is not full to add fish onto it
				pthread_cond_wait(&nonFull[info->id], &lock[info->id]);
			}
			// add to buffer
			printf("# begin produce\n");
			buffer[info->id].stock[buffer[info->id].count+1]=workee[buffer[info->id].count-1];// add fish object to buffer
			buffer[info->id].count+=1;
			
			printf("# end produce\n");
			pthread_cond_signal(&nonEmpty[info->id]);
			pthread_mutex_unlock(&lock[info->id]);
		}
		else {
			printf("producer: last worker, finish\n");
			// if this is the last thread, then fish is almost entirely processed
			// last step
				printf("value  :%d\n ",buffer[info->id].count);		
				workee[buffer[info->id].count-1]->Tend= get_time_sec();
				printf("Fish  %d is done \n",fish_count);
		}
		
		// break statement
		fish_count+=1;
		if(fish_count>NUM_FISH){
			break;
		}
		
	}
	return NULL;
}








