/*	Mkhanyisi Gamedze
* 	CS336 : Parallel & distributed processing
* 	my_math.c:
*	math routines to support circadian simulation code
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Return length of vector (x,y)
float norm(float x, float y) {
  return sqrt( pow(x,2) + pow(y,2) );
} // end norm

// swap the floats a and b
void swapScalar(float *a, float *b) {
  float tmp;
  tmp = *a; // stores the value of a
  *a = *b; // the new value of a is the value of b
  *b = tmp; // the value of b is tmp
}

// return the Euclidean distance from (x1,y1) to (x2,y2)
float distance(float x1, float y1, float x2, float y2){
	float distance;
	//printf("\nvalues  %f   %f   %f  %f\n",x1, y1, x2,y2);
    distance = sqrt((y2-y1) * (y2-y1) + (x2 - x1) * (x2 - x1));
    return distance;
}

// Return 1 if n is odd. Return 0 if n is not odd.
int isOdd( int n ){
	if(n%2==1){
		printf(" %d   is Odd \n", n);	
		return 1;
	}
	else{
		printf(" %d    is Even\n", n);
		return 0; // even number
	}
}
// return 1 if n is even. Return 0 if n is not even.
int isEven( int n ){
	if(n%2==1){
		printf("  %d  is Odd\n", n);
		return 0; // is odd
	}
	else{
		printf("  %d   is Even\n",n);
		return 1; // is even
	}
}


// return the number of Nans in the array.
int countNans(float *array, int N){
	// figure this one out
	int i,numNans=0;
        for(i = 1; i < N; i++){
		if(isnan(array[i])){
			numNans+=1;
		}
	}
	return numNans;
}
    

// Return the maxium value in the array (of length N)
 float max(float *array, int N){
	float largest = array[0];
 	int i;
        for(i = 1; i < N; i++){
		if (largest < array[i]){
			largest = array[i];
		}
	}
	return (float)largest;
}
    

// Return the sum of the N floats in the array.
 float sum(float *array, int N){
	float sum = array[0];
 	int i;
        for ( i = 1; i < N; i++){
			sum+= array[i];
	}
	return sum;
}


// Return the sum of the N floats in the array.
 float mean(float *array, int N){
	float sum = array[0];
 	int i;
        for (i = 1; i < N; i++){
			sum+= array[i];
	}
	int mean=sum/N;
	return mean; 
}

 // Return the standard deviation of N floats in the array.
    // Note: This uses the Bessel correction.
 float std(float *array, int N){
	float sum = array[0];
 	float standardDeviation=0.0;
        int i;
	 for (i = 1; i < N; i++){
			sum+= array[i];
	}
	int mean=sum/N;
	for(i=0; i<N; ++i)
        standardDeviation += pow(array[i] - mean, 2);

    	return sqrt(standardDeviation/N);	
}

 float printArray(float *array, int N){
	int i;
	printf("\n");
	for (i = 0; i < N; i++){
			printf(" %f ", array[i]);
	}
	printf("\n");
    	return 0;	
}

// swap the pointers a and b
void swapArrays(float **a, float **b){
    float *c=*a;
    *a=*b;
    *b=c;
}


float** add_arr(float **m1,float **m2,int row,int col){
	// assume the two matrices are of same size
    float** m3=(float**)malloc(sizeof(float*)*row*col);
    //float m3[row][col];
    int i,j;
    for(i=1;i<=row;i++){
    	for(j=1;j<=col;j++){
		printf("  %f      %f\n",m1[i][j],m2[i][j]);
    		m3[i][j] =  (m1[i][j] + m2[i][j]);
    	}
    }
	return m3;
}


float *createAscendingArray(int N) {
  float *ret = malloc(sizeof(float)*N);
  int i;
  for (i=0; i<N; i++)
    ret[i] = (float)i;
  return ret;
}

float dot_product(float v[], float u[], int n){
	int i;
	float result=0.0;
	for (i = 0; i < n; i++){
		result += v[i]*u[i];
	}
	return result;
}


float randFloat() {
 	float rand_max=RAND_MAX;
	return (float)rand()/rand_max;
}



