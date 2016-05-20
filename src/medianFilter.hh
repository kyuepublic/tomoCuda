/*
THis is the class for median filter

*/
#ifndef MEDIANFILTER_H
#define MEDIANFILTER_H

#define BLOCK_X  8
#define BLOCK_Y  8



class medianFilter {
  // pointer to the GPU memory where the array is stored
  float* array_device_in;
  float* array_device_out;
  // pointer to the CPU memory where the array is stored
  float* array_host;

//  float* v_device;
  // length of the array (number of elements)
  int inlength;
  int outlength;
  int nx;
  int ny;
  int filterSize;

public:
  /*

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  medianFilter(float* INPLACE_ARRAY1, int nx_, int ny_, int filterSize_); // constructor (copies to GPU)


  ~medianFilter(); // destructor

  void runFilter(int size); // does operation inplace on the GPU

  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void retreive_to (float* INPLACE_ARRAY1);


};

#endif