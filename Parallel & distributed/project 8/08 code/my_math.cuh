/* my_math.h:
 * math routines to support circadian simulation code
 */

#ifndef MY_MATH_H

#define MY_MATH_H

// Return length of vector (x,y)
__device__  __host__  float norm(float x, float y);

// return the distance from (x1,y1) to (x2,y2)
__device__  __host__  float distance(float x1, float y1, float x2, float y2);

// swap the scalars
__device__  __host__  void swapScalar(float *a, float *b);

// return the number of Nans in the array.
__device__  __host__  int countNans(float *array, int N);

// Return the sum of the N floats in the array.
__device__  __host__  float sum(float *array, int N);

// Return the mean value of N floats in the array.
__device__  __host__  float mean(float *array, int N);

// Return the standard deviation of N floats in the array.
__device__  __host__  float stdev(float *array, int N);

// Return the minimium value in the array (of length N)
__device__  __host__  float min(float *array, int N);

// Return the maxium value in the array (of length N)
__device__  __host__  float max(float *array, int N);

// swap the pointers a and b
__device__  __host__  void swap(float **a, float **b);

// Return a non-zero value if n is odd, 0 otherwise
__device__  __host__  int isOdd( int n );

// Return a non-zero value  if n is even, 0 otherwise
__device__  __host__  int isEven( int n );

#endif
