#include <medianKernel.cu>
#include <medianFilter.hh>
#include <assert.h>
//#include <time.h>
#include <sys/time.h>
#include <iostream>


using namespace std;


medianFilter::medianFilter (/*float* array_host_,*/ int nx_, int ny_, int nz_, int filterSize_)
{

// the matrix size
  nx = nx_;
  ny = ny_;
  nz = nz_;

  filterSize = filterSize_;

  // inlength is the 2d image size of each image
  // outlength is the 2d image size of each image
//  inlength=(nx+filterSize-1)*(ny+filterSize-1)*nz;
//  outlength = nx*ny*nz;

  // inlength is the 2d image size of each image
  inlength=(nx+filterSize-1)*(ny+filterSize-1);
  outlength = nx*ny;


//  for (int i =0; i< inlength; i++)
//    printf("the element is %f \n", array_host[i]);

  insize = inlength * sizeof(float);
  outsize = outlength * sizeof(float);

//  if ((array_host_out = (float*)malloc(sizeof(float)*length))) == 0)
//  {
//    fprintf(stderr,"malloc1 Fail \n");
//    return;
//  }

  cudaError_t err = cudaMalloc((void**) &array_device_in, insize);
  assert(err == 0);

  err = cudaMalloc((void**) &array_device_out, outsize);
  assert(err == 0);

//  err = cudaMemcpy(array_device_in, array_host, insize, cudaMemcpyHostToDevice);
//  assert(err == 0);



}

medianFilter::~medianFilter()
{

//  cudaFree(v_device);

  cudaFree(array_device_in);
  cudaFree(array_device_out);

}

void medianFilter::run2DFilter(int size)
{

//  cudaError_t err0 = cudaMalloc((void**) &v_device, size*size*sizeof(float));
//  cudaMemset(v_device, 0, size*size*sizeof(float));
//  double iStart = cpuSecond();

  int block_size_x = BLOCK_X;
  int block_size_y = BLOCK_Y;

  dim3 blocks((nx+block_size_x-1)/block_size_x, (ny+block_size_y-1)/block_size_y);
  dim3 threads(block_size_x,block_size_y);

//  af_border_type pad = AF_PAD_SYM;
//  Param<float> out;
//  out.ptr=array_device_out;
//  out.dims[0]=


  switch(filterSize)
  {
    case 2:
      kernel2ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 3:
      kernel3ME <<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 4:
      kernel4ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 5:
      kernel5ME <<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 6:
      kernel6ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 7:
      kernel7ME <<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 8:
      kernel8ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 9:
      kernel9ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 10:
      kernel10ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 11:
      kernel11ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 12:
      kernel12ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 13:
      kernel13ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 14:
      kernel14ME<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    case 15:
      kernel15ME <<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
      break;
    default:
      break;

  }

  // add these to synchronzie the thread
  cudaDeviceSynchronize();

//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));

//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//  gpuErrchk( cudaPeekAtLastError());
//  gpuErrchk( cudaDeviceSynchronize() );

}

void medianFilter::run2DRemoveOutliner(int size, int diff)
{



//  double iStart = cpuSecond();

  int block_size_x = BLOCK_X;
  int block_size_y = BLOCK_Y;

  dim3 blocks((nx+block_size_x-1)/block_size_x, (ny+block_size_y-1)/block_size_y);
  dim3 threads(block_size_x,block_size_y);


  switch(filterSize)
  {
    case 2:
      reomveOutliner2D2ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 3:
      reomveOutliner2D3ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 4:
      reomveOutliner2D4ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 5:
      reomveOutliner2D5ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 6:
      reomveOutliner2D6ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 7:
      reomveOutliner2D7ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 8:
      reomveOutliner2D8ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 9:
      reomveOutliner2D9ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 10:
      reomveOutliner2D10ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 11:
      reomveOutliner2D11ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 12:
      reomveOutliner2D12ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 13:
      reomveOutliner2D13ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 14:
      reomveOutliner2D14ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    case 15:
//      reomveOutliner2D15<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
//      reomveOutliner2D15M<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      reomveOutliner2D15ME<<<blocks,threads>>>(nx, ny, diff, array_device_out, array_device_in);
      break;
    default:
      break;

  }


  // add these to synchronzie the thread
  cudaDeviceSynchronize();

//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));

  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void medianFilter::retreive()
{

//  int outsize = outlength * sizeof(float);
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
//  time_t start = time(NULL);

//  double iStart = cpuSecond();

  cudaMemcpy(array_host_, array_device_out, outsize, cudaMemcpyDeviceToHost);

//  printf("total copy back time for this process took %f sec \n",(cpuSecond() - iStart));

  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}


void medianFilter::setImage(float* array_host_)
{

//  double iStart = cpuSecond();

  array_host = array_host_;

  cudaError_t err = cudaMemcpy(array_device_in, array_host, insize, cudaMemcpyHostToDevice);

//  printf("total copy to device time for this process took %f sec \n",(cpuSecond() - iStart));
  gpuErrchk( cudaPeekAtLastError());
  assert(err == 0);

}

double medianFilter::cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

//void medianFilter::run3DRemoveOutliner(int size, int diff)
//{
//
//  double iStart = cpuSecond();
//
//  int block_size_x = BLOCK_X;
//  int block_size_y = BLOCK_Y;
//  int block_size_z = BLOCK_Z;
//
//  dim3 gridSize(((nx+block_size_x-1)/block_size_x), ((ny+block_size_y-1)/block_size_y), ((nz+block_size_z-1)/block_size_z));
//  dim3 blockSize(block_size_x, block_size_y, block_size_z);
//
//
//  switch(filterSize)
//  {
//    case 2:
//      reomveOutliner3D2<<<gridSize,blockSize>>>(nx, ny, nz, diff, array_device_out, array_device_in);
//      break;
////    case 3:
////      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 4:
////      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 5:
////      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 6:
////      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
//    case 15:
//      reomveOutliner3D15<<<gridSize,blockSize>>>(nx, ny, nz, diff, array_device_out, array_device_in);
//      break;
//    default:
//      break;
//
//  }
//
//
//  // add these to synchronzie the thread
////  cudaDeviceSynchronize();
//
//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));
//
//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//}


//void medianFilter::run3DFilter(int size)
//{
//
//  double iStart = cpuSecond();
//
//  int block_size_x = BLOCK_X;
//  int block_size_y = BLOCK_Y;
//  int block_size_z = BLOCK_Z;
//
//  dim3 gridSize(((nx+block_size_x-1)/block_size_x), ((ny+block_size_y-1)/block_size_y), ((nz+block_size_z-1)/block_size_z));
//  dim3 blockSize(block_size_x, block_size_y, block_size_z);
//
//  switch(filterSize)
//  {
//    case 2:
//      kernel3D2<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
////    case 3:
////      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 4:
////      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 5:
////      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 6:
////      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
//    case 15:
//      kernel3D15<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
//    default:
//      break;
//
//  }
//
//  gpuErrchk( cudaPeekAtLastError());
//  gpuErrchk( cudaDeviceSynchronize() );
//
//  // add these to synchronzie the thread
////  cudaDeviceSynchronize();
//
//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));
//
////  cudaError_t err = cudaGetLastError();
////  assert(err == 0);
//}

//void medianFilter::run3DFilterXZ(int size)
//{
//
//  double iStart = cpuSecond();
//
//  int block_size_x = BLOCK_X;
//  int block_size_y = BLOCK_Y;
//  int block_size_z = BLOCK_Z;
//
//
//
//  int tmp =  nz;
//  nz = ny;
//  ny = tmp;
//
//  dim3 gridSize(((nx+block_size_x-1)/block_size_x), ((ny+block_size_y-1)/block_size_y), ((nz+block_size_z-1)/block_size_z));
//  dim3 blockSize(block_size_x, block_size_y, block_size_z);
//
//
//
//  switch(filterSize)
//  {
//    case 2:
//      kernel3D2<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
////    case 3:
////      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 4:
////      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 5:
////      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 6:
////      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
//    case 15:
////      kernel3D15XZ<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      kernel3D15XZME<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
//    default:
//      break;
//
//  }
//
//
//  gpuErrchk( cudaPeekAtLastError());
//  gpuErrchk( cudaDeviceSynchronize() );
//
//  // add these to synchronzie the thread
////  cudaDeviceSynchronize();
//
//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));
//
////  cudaError_t err = cudaGetLastError();
////  assert(err == 0);
//}

//void medianFilter::run2DLoopFilter(int size)
//{
//
//  double iStart = cpuSecond();
//
//
//  int block_size_x = BLOCK_X;
//  int block_size_y = BLOCK_Y;
//
//  dim3 gridSize((nx+block_size_x-1)/block_size_x, (ny+block_size_y-1)/block_size_y);
//  dim3 blockSize(block_size_x,block_size_y);
//
//  switch(filterSize)
//  {
//    case 2:
//      kernel3D2<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
////    case 3:
////      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 4:
////      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 5:
////      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 6:
////      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
//    case 15:
//      kernelLool3D15<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
//    default:
//      break;
//
//  }
//  // add these to synchronzie the thread
//  cudaDeviceSynchronize();
//
//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));
//
//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//}

//void medianFilter::run2DLoopFilterXZY(int size)
//{
//
//  double iStart = cpuSecond();
//  int block_size_x = BLOCK_X;
//  int block_size_y = BLOCK_Y;
//
//  dim3 gridSize((nx+block_size_x-1)/block_size_x, (nz+block_size_y-1)/block_size_y);
//  dim3 blockSize(block_size_x,block_size_y);
//
//  switch(filterSize)
//  {
//    case 2:
//      kernel3D2<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
////    case 3:
////      kernel3<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 4:
////      kernel4<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 5:
////      kernel5<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
////    case 6:
////      kernel6<<<blocks,threads>>>(nx, ny, array_device_out, array_device_in);
////      break;
//    case 15:
//      kernelLool3D15XZY<<<gridSize,blockSize>>>(nx, ny, nz, array_device_out, array_device_in);
//      break;
//    default:
//      break;
//
//  }
//  // add these to synchronzie the thread
//  cudaDeviceSynchronize();
//
//  printf("total execution time for this kernel took %f sec \n",(cpuSecond() - iStart));
//
//  cudaError_t err = cudaGetLastError();
//  assert(err == 0);
//}
