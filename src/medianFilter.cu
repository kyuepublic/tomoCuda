#include <medianKernel.cu>
#include <medianFilter.hh>
#include <assert.h>
#include <iostream>

using namespace std;

medianFilter::medianFilter (float* array_host_, int nx_, int ny_, int filterSize_)
{

  array_host = array_host_;
  nx = nx_;
  ny = ny_;
  filterSize = filterSize_;



  inlength=(nx+filterSize-1)*(ny+filterSize-1);
  outlength = nx*ny;

//  for (int i =0; i< inlength; i++)
//    printf("the element is %f \n", array_host[i]);

  int insize = inlength * sizeof(float);
  int outsize = outlength * sizeof(float);

//  if ((array_host_out = (float*)malloc(sizeof(float)*length))) == 0)
//  {
//    fprintf(stderr,"malloc1 Fail \n");
//    return;
//  }

  cudaError_t err = cudaMalloc((void**) &array_device_in, insize);
  err = cudaMalloc((void**) &array_device_out, outsize);

  assert(err == 0);
  err = cudaMemcpy(array_device_in, array_host, insize, cudaMemcpyHostToDevice);
  assert(err == 0);
}

void medianFilter::runFilter(int size)
{


//  kernel_add_one<<<64, 64>>>(array_device, length);

//  cudaError_t err0 = cudaMalloc((void**) &v_device, size*size*sizeof(float));
//  cudaMemset(v_device, 0, size*size*sizeof(float));

  int block_size_x = BLOCK_X;
  int block_size_y = BLOCK_Y;

  dim3 blocks((nx+block_size_x-1)/block_size_x, (ny+block_size_y-1)/block_size_y);
  dim3 threads(block_size_x,block_size_y);

  switch(filterSize)
  {
    case 2:
      kernel2<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 3:
      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 4:
      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 5:
      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 6:
      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 15:
      kernel15<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    default:
      break;

  }

  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void medianFilter::retreive()
{



  int outsize = outlength * sizeof(float);
  cudaMemcpy(array_host, array_device_out, outsize, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0)
  {
    cout << err << endl; assert(0);
  }

}

void medianFilter::retreive_to (float* array_host_)
{
//  assert(length == length_);

  int outsize = outlength * sizeof(float);
  cudaMemcpy(array_host_, array_device_out, outsize, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

medianFilter::~medianFilter()
{

//  cudaFree(v_device);

  cudaFree(array_device_in);
  cudaFree(array_device_out);

}
