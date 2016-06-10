/*
THis is the class for median filter

*/
#ifndef MEDIANFILTER_H
#define MEDIANFILTER_H

#define BLOCK_X  8
#define BLOCK_Y  8
#define BLOCK_Z  8



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
  int nz;
  int filterSize;

  int insize;
  int outsize;

public:
  /*

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  medianFilter(/*float* INPLACE_ARRAY1,*/ int nx_, int ny_, int nz_, int filterSize_); // constructor (copies to GPU)


  ~medianFilter(); // destructor

  void run2DFilter(int size); // does operation inplace on the GPU

  void run3DFilter(int size); // does operation inplace on the GPU

  void run2DLoopFilter(int size);

  void run3DRemoveOutliner(int size, int diff);

  void run2DLoopFilterXZY(int size);

  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void retreive_to(float* array_host_);

  // Set image
  void setImage(float* array_host_);

  double cpuSecond();

};

#endif