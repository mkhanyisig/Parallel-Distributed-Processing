/* test_my_math.c
 * Tests all routines in my_math.h
 * Stephanie Taylor
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "my_math.h"

/* Functions that test each my_math function, printing the
 * result to the screen
 */

void test_norm() {
  printf("\nTesting Norm\n");
  float n = norm(0.1, 0.1);
  printf("norm(0.1,0.1) is %f, and should be 0.141421\n",n);
  n = norm(0.0,0.0);
  printf("norm(0.0,0.0) is %f, and should be 0.000000\n",n);
  n = norm(-1.0,0.1);
  printf("norm(-1.0,0.1) is %f, and should be 1.004988\n",n);
}

void test_swapScalar() {
  float a = 5.0;
  float b = 6.0;
  printf("Before swap, a=%f and b=%f\n",a,b);
  swapScalar(&a,&b);
  printf("After swap, a=%f and b=%f\n",a,b);
}

void test_distance(){
	// distance between (0,0) & (5,7)
	printf("\n**testing distance between (0,0) & (5,7)\n");
	printf("expected answer  8.602    ");
	float dist=distance(0,0,5,7);
	printf("computed distance.  =%f\n\n",dist);	
}

void test_Odd(){
	// test  1 & 2 for oddness
	printf("Testing isOdd\n");
	isOdd((int)1);
	isOdd((int)2);
}

void test_Even(){
	// test  1 & 2 for oddness
	printf("Testing isEven\n");
	isEven((int)1);
	isEven((int)2);
}

void test_numNans(){
	printf("testing numNans\n");
	float a[] = {1.0, 2.0, 3.0,NAN, 8.0, 3.0,NAN, 5.0,0.0, 56.1};// 10 elements
	int num=countNans(a,10);
	printf("  2 Null's in list a, reported : %d\n",num);

}


void test_Max(){
	printf("testing Max\n");
	float a[] = {1.0, 2.0, 3.0, 8.0, 3.0,26.0, 5.0,0.0, 56.1,44.0};// 10 elements
	int num=max(a,10);
	printf("  Max of a : %d\n",num);

}

void test_Sum(){
	printf("testing Sum\n");
	float a[] = {1.0, 2.0, 3.0, 8.0, 3.0,26.0, 5.0,0.0, 56.1,44.0};// 10 elements
	int num=sum(a,10);
	printf("  Sum of a : %d\n",num);

}

void test_Mean(){
	printf("testing Mean\n");
	float a[] = {1.0, 2.0, 3.0, 8.0, 3.0,26.0, 5.0,0.0, 56.1,44.0};// 10 elements
	int num=mean(a,10);
	printf("  Mean of a : %d\n",num);

}

void test_Std(){
	printf("testing STD\n");
	float a[] = {1.0, 2.0, 3.0, 8.0, 3.0,26.0};// 10 elements
	int num=std(a,10);
	printf("  STD of a : %d\n",num);
}


void test_Swap(){
	printf("testing STD\n");
	float a[] = {1.0, 2.0, 3.0, 8.0, 3.0,26.0};// 10 elements
	float b[] = {9.0, 8.0, 5.0,0.0, 56.1,44.0};
	printf("before swap");
	printArray(a,6);
	printArray(b, 6);
	swapArrays((float**)&a,(float**)&b);
	printf("after swap");
	printArray(a,6);
	printArray(b, 6);
}


/*  // please help here
void test_addArr(){
	printf("testing Add Array\n");
	float a[2][4]= { {10, 11, 12, 13},{14, 15, 16, 17}};
	float b[2][4]= { {1, 2, 3, 4},{5, 6, 7, 8}};	
	float c;// should be [2][4] array as well
	c=add_arr(&a,&b,2,4);
	printf("after Adding arrays");
	//printArray(c,8);
}
*/

void test_dotProduct(){
	printf("testing dot product\n");
	float a[]=  {10, 11, 12, 13};

	float b[]=  {1, 2, 3, 4};
	printf("before a.b\n");
	printArray(a,4);
	printArray(b, 4);
	printf("The dot product is %f  \n",dot_product(a,b,4));
}


void testRand(){
	float a=randFloat();
	printf("random float generated  %f\n",a);
}



/* Main function */
// Just comment out any line that runs a test you don't want to run.
int main(int argc, char* argv[]) {
  test_norm(); 
  test_swapScalar();
  test_distance();
  test_Odd();
  test_Even();
  test_numNans();
  test_Max();
  test_Sum();
  test_Mean();
  test_Std();
  test_Swap();
  //test_addArr();
  test_dotProduct();
  testRand();
} // main

