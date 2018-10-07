#include<stdio.h>
/* reference: https://www.codingunit.com/c-tutorial-binary-file-io */	
struct rec{
	int x,y,z;
};

int main(){
	int counter;
	FILE *ptr_myfile;
	struct rec my_record;

	ptr_myfile=fopen("test.bin","rb");// read binary file we just wrote
	if (!ptr_myfile){
		printf("Unable to open file!");
		return 1;
	}
	for ( counter=1; counter <= 10; counter++){// read all ten lines
		//  With the fread we read-in the records (one by one)
		fread(&my_record,sizeof(struct rec),1,ptr_myfile);
		printf("%d\n",my_record.x);
	}
	fclose(ptr_myfile);
	return 0;
}
