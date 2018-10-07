/*	Mkhanyisi Gamedze
* 	CS336 : Parallel & distributed processing
* 	writebinary.c
*/

#include<stdio.h>
/* reference: https://www.codingunit.com/c-tutorial-binary-file-io */
struct rec{
	int x,y,z;
};

int main(){
	int counter;
	FILE *ptr_myfile;
	struct rec my_record;

	ptr_myfile=fopen("test.bin","wb");// open file in write binary mode
	if (!ptr_myfile){//check if the file is open, if not, display error
		printf("Unable to open file!");
		return 1;
	}
	for ( counter=1; counter <= 10; counter++){//creating ten records
		my_record.x= counter;
		fwrite(&my_record, sizeof(struct rec), 1, ptr_myfile);
	}
	fclose(ptr_myfile);
	return 0;
}
