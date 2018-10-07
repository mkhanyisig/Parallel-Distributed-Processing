/* my_math.h:
 * math routines to support circadian simulation code
 */

#ifndef MY_MATH_H

#define MY_MATH_H

// Return length of vector (x,y)
float norm(float x, float y);

// return the distance from (x1,y1) to (x2,y2)
float distance(float x1, float y1, float x2, float y2);

// swap the scalars
void swapScalar(float *a, float *b);

// return the number of Nans in the array.
int countNans(float *array, int N);

// Return the sum of the N floats in the array.
float sum(float *array, int N);

// Return the mean value of N floats in the array.
float mean(float *array, int N);

// Return the standard deviation of N floats in the array.
float std(float *array, int N);

// Return the minimium value in the array (of length N)
float min(float *array, int N);

// Return the maxium value in the array (of length N)
float max(float *array, int N);

// swap the pointers a and b
void swap(float **a, float **b);

// Return a non-zero value if n is odd, 0 otherwise
int isOdd( int n );

// Return a non-zero value  if n is even, 0 otherwise
int isEven( int n );

#endif
