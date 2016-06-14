// Only support reflect mode right now

#include <stdio.h>
#include <medianFilter.hh>

#define IN(X,Y)  d_in[X+Y*(14+nx)]
//
//v[i++] = d_in[xx+yy*newnx+zoffset];

// various windows size
//__global__ void kernel(int nx, int ny, float *d_out, float *d_in, int size)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int i = 0;
////    float v[9] = {0,0,0,0,0,0,0,0,0}; // zero padding
//
//    int offset = (size-1)/2;
//    int winSize = size*size;
//    int winOffset = (winSize-1)/2;
//
//    float v[winSize] = {0};
//
//
//    for (int xx = x - offset; xx <= x + offset; xx++) {
//        for (int yy = y - offset; yy <= y + offset; yy++) {
//            if (0 <= xx && xx < nx && 0 <= yy && yy < ny) // boundaries
//
//                v[i++] = d_in[yy*nx + xx];
//        }
//    }
//
//    // bubble-sort
//    for (int i = 0; i < winSize; i++) {
//        for (int j = i + 1; j < winSize; j++) {
//            if (v[i] > v[j]) { /* swap? */
//                float tmp = v[i];
//                v[i] = v[j];
//                v[j] = tmp;
//            }
//        }
//    }
//
////     printf("the x not is %d, y is %d, result is %f \n", x, y, v[4] );
//    // pick the middle one
//    d_out[y*nx + x] = v[winOffset];
//}


// window 2 by 2
__global__ void kernel2(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int winSize = 2;
    float v[4] = {0};

    int vecSize = winSize*winSize;
    int loffset = winSize/2;
    int roffset = winSize/2 - 1;
    int toffset = loffset+roffset;

    x = x + loffset;
    y = y + loffset;



    int i = 0;


    for (int xx = x - loffset; xx <= x + roffset; xx++)
    {
        for (int yy = y - loffset; yy <= y + roffset; yy++)
        {
            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset)
             {// boundaries
                v[i++] = d_in[yy*(nx+toffset) + xx];
             }
        }
    }

    // bubble-sort
    for (int i = 0; i < vecSize; i++)
    {
        for (int j = i + 1; j < vecSize; j++)
        {
            if (v[i] > v[j])
            { /* swap? */
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    // pick the middle one
    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];

}


// window 3 by 3
__global__ void kernel3(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int winSize = 3;
    float v[9] = {0};

    int vecSize = winSize*winSize;
    int loffset = winSize/2;
    int roffset = (winSize-1)/2;
    int toffset = loffset+roffset;

    x = x + loffset;
    y = y + loffset;

    int i = 0;

    for (int xx = x - loffset; xx <= x + roffset; xx++)
    {
        for (int yy = y - loffset; yy <= y + roffset; yy++)
        {
            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                v[i++] = d_in[yy*(nx+toffset) + xx];
        }
    }

    // bubble-sort
    for (int i = 0; i < vecSize; i++)
    {
        for (int j = i + 1; j < vecSize; j++)
        {
            if (v[i] > v[j])
            { /* swap? */
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    // pick the middle one
    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
}

// windows size 4 byb 4
__global__ void kernel4(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int winSize = 4;
    int loffset = winSize/2;
    int roffset = winSize/2 - 1;
    int toffset = loffset+roffset;

    x = x + loffset;
    y = y + loffset;

    int i = 0;
    float v[16] = {0};

    for (int xx = x - loffset; xx <= x + roffset; xx++)
    {
        for (int yy = y - loffset; yy <= y + roffset; yy++)
        {
            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                v[i++] = d_in[yy*(nx+toffset) + xx];
        }
    }

    // bubble-sort
    for (int i = 0; i < 16; i++)
    {
        for (int j = i + 1; j < 16; j++)
        {
            if (v[i] > v[j])
            { /* swap? */
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    // pick the middle one
    d_out[(y-loffset)*nx + x-loffset] = v[8];
}

// Windows size 5 by b5
__global__ void kernel5(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int winSize = 5;
    float v[25] = {0};

    int vecSize = winSize*winSize;
    int loffset = winSize/2;
    int roffset = (winSize-1)/2;
    int toffset = loffset+roffset;

    x = x + loffset;
    y = y + loffset;

    int i = 0;

    for (int xx = x - loffset; xx <= x + roffset; xx++)
    {
        for (int yy = y - loffset; yy <= y + roffset; yy++)
        {
            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                v[i++] = d_in[yy*(nx+toffset) + xx];
        }
    }

    // bubble-sort
    for (int i = 0; i < vecSize; i++)
    {
        for (int j = i + 1; j < vecSize; j++)
        {
            if (v[i] > v[j])
            { /* swap? */
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    // pick the middle one
    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];

}

// windows size 6 byb 6
__global__ void kernel6(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int winSize = 6;
    float v[36] = {0};

    int vecSize = winSize*winSize;
    int loffset = winSize/2;
    int roffset = winSize/2 - 1;
    int toffset = loffset+roffset;

    x = x + loffset;
    y = y + loffset;

    int i = 0;


    for (int xx = x - loffset; xx <= x + roffset; xx++)
    {
        for (int yy = y - loffset; yy <= y + roffset; yy++)
        {
            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                v[i++] = d_in[yy*(nx+toffset) + xx];
        }
    }

    // bubble-sort
    for (int i = 0; i < vecSize; i++)
    {
        for (int j = i + 1; j < vecSize; j++)
        {
            if (v[i] > v[j])
            { /* swap? */
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    // pick the middle one
    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
}


// window size 15 by b15
__global__ void kernel15(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 15;
        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

        int i = 0;

        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
//                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                    v[i++] = d_in[yy*(nx+toffset) + xx];
            }
        }

        // bubble-sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                { /* swap? */
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
    }

}

__global__ void kernel3D2(int nx, int ny, int nz,  float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

    if ((x < nx) && (y < ny) && (z < nz))
    {
        int winSize = 2;
        float v[4] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;
        int newnx=toffset+nx;
        int newny=toffset+ny;

        x = x + loffset;
        y = y + loffset;

        int i = 0;

        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
//                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
            }
        }

        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                { /* swap? */
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];

    }

}

__global__ void kernel3D15(int nx, int ny, int nz,  float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

//    int offset = x+y* nx + ny * nx * z;
    if ((x < nx) && (y < ny) && (z < nz))
    {
        // initial the window size, the local vector size
        int winSize = 15;
        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2; // the left and top offset
        int roffset = (winSize-1)/2; // the right and bottom offset
        int toffset = loffset+roffset; // the overall offset

// The new x' y' is the plus offset
        x = x + loffset;
        y = y + loffset;

        int i = 0;
        // Put the neighbour pixel into the local memory for the later bubble sort
        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
                    // find the read address of the x y z pixel
                    v[i++] = d_in[xx+yy*(nx+toffset)+z*(nx+toffset)*(ny+toffset)];
            }
        }

        // do the bubble sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                {   // bubble sort
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        //    printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );
        // put the final result value to the output array
        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];

    }

}


__global__ void kernelLool3D15(int nx, int ny, int nz,  float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

//    int offset = x+y* nx + ny * nx * z;
    if ((x < nx) && (y < ny))
    {
        int winSize = 15;
        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

        int i = 0;



        for(int z = 0; z < nz; z++)
        {
            i = 0;

            for (int xx = x - loffset; xx <= x + roffset; xx++)
            {
                for (int yy = y - loffset; yy <= y + roffset; yy++)
                {

                    v[i++] = d_in[xx+yy*(nx+toffset)+z*(nx+toffset)*(ny+toffset)];
                }
            }

            for (int i = 0; i < vecSize; i++)
            {
                for (int j = i + 1; j < vecSize; j++)
                {
                    if (v[i] > v[j])
                    { /* swap? */
                        float tmp = v[i];
                        v[i] = v[j];
                        v[j] = tmp;
                    }
                }
            }

//            printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );

            d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];

        }

    }

}

__global__ void kernelLool3D15XZY(int nx, int ny, int nz,  float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int z = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

//    int offset = x+y* nx + ny * nx * z;
    if ((x < nx) && (z < nz))
    {
        int winSize = 15;
        float v[225]={0};


        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        int newnx = nx+toffset;
        int newny = ny+toffset;
        int zoffset = z*newnx*newny;

        x = x + loffset;
//        y = y + loffset;
//        z = z + loffset;

        int i = 0;

        for(int y = loffset; y < ny+loffset; y++)
        {
            i = 0;

//            for (int xx = x - loffset; xx <= x + roffset; xx++)
//            {
//                for (int yy = y - loffset; yy <= y + roffset; yy++)
//                {
//
//                    v[i++] = d_in[xx+yy*newnx+zoffset];
//                }
//            }



            for (int i = 0; i < vecSize; i++)
            {
                for (int j = i + 1; j < vecSize; j++)
                {
                    if (v[i] > v[j])
                    {
                        float tmp = v[i];
                        v[i] = v[j];
                        v[j] = tmp;
                    }
                }
            }

//            printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );

            d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];

        }

    }

}

__global__ void reomveOutliner3D2(int nx, int ny, int nz, int diff, float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

//    int offset = x+y* nx + ny * nx * z;
    if ((x < nx) && (y < ny) && (z < nz))
    {
        int winSize = 2;
        float v[4] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;
        int newnx=toffset+nx;
        int newny=toffset+ny;

        x = x + loffset;
        y = y + loffset;

        int i = 0;

        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
//                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
            }
        }

        // get the current pixel value
        // TODO get from local buffer instead of global memory

        float currentPixel = d_in[x+y*newnx+z*newnx*newny];



        // More optimize for the bubble sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                { /* swap? */
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        // TODO more optimize for this part
        int mask = 0;
        float realdiff = currentPixel-v[vecSize/2];
        printf("the x is %d, y is %d, z is %d, current is %f, result is %f \n", x, y, z, currentPixel, v[vecSize/2] );

        if( realdiff >= diff)
            mask = 1;
        else
            mask = 0;



        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2]*mask+currentPixel*(1-mask);

    }

}

__global__ void reomveOutliner2D15(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 15;
        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

        int i = 0;

        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
                v[i++] = d_in[yy*(nx+toffset) + xx];
            }
        }

        float currentPixel = IN(x, y);

        // bubble-sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                {
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        int mask = 0;
        if((currentPixel-v[vecSize/2]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2]*mask+currentPixel*(1-mask);
    }

}

__global__ void reomveOutliner2D15M(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 15;
//        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

//        int i = 0;

        float v[225]={IN(x-7, y-7), IN(x-7, y-6), IN(x-7, y-5), IN(x-7, y-4), IN(x-7, y-3), IN(x-7, y-2), IN(x-7, y-1), IN(x-7, y), IN(x-7, y+1), IN(x-7, y+2), IN(x-7, y+3), IN(x-7, y+4), IN(x-7, y+5), IN(x-7, y+6), IN(x-7, y+7),
                      IN(x-6, y-7), IN(x-6, y-6), IN(x-6, y-5), IN(x-6, y-4), IN(x-6, y-3), IN(x-6, y-2), IN(x-6, y-1), IN(x-6, y), IN(x-6, y+1), IN(x-6, y+2), IN(x-6, y+3), IN(x-6, y+4), IN(x-6, y+5), IN(x-6, y+6), IN(x-6, y+7),
                      IN(x-5, y-7), IN(x-5, y-6), IN(x-5, y-5), IN(x-5, y-4), IN(x-5, y-3), IN(x-5, y-2), IN(x-5, y-1), IN(x-5, y), IN(x-5, y+1), IN(x-5, y+2), IN(x-5, y+3), IN(x-5, y+4), IN(x-5, y+5), IN(x-5, y+6), IN(x-5, y+7),
                      IN(x-4, y-7), IN(x-4, y-6), IN(x-4, y-5), IN(x-4, y-4), IN(x-4, y-3), IN(x-4, y-2), IN(x-4, y-1), IN(x-4, y), IN(x-4, y+1), IN(x-4, y+2), IN(x-4, y+3), IN(x-4, y+4), IN(x-4, y+5), IN(x-4, y+6), IN(x-4, y+7),
                    IN(x-3, y-7), IN(x-3, y-6), IN(x-3, y-5), IN(x-3, y-4), IN(x-3, y-3), IN(x-3, y-2), IN(x-3, y-1), IN(x-3, y), IN(x-3, y+1), IN(x-3, y+2), IN(x-3, y+3), IN(x-3, y+4), IN(x-3, y+5), IN(x-3, y+6), IN(x-3, y+7),
                    IN(x-2, y-7), IN(x-2, y-6), IN(x-2, y-5), IN(x-2, y-4), IN(x-2, y-3), IN(x-2, y-2), IN(x-2, y-1), IN(x-2, y), IN(x-2, y+1), IN(x-2, y+2), IN(x-2, y+3), IN(x-2, y+4), IN(x-2, y+5), IN(x-2, y+6), IN(x-2, y+7),
                    IN(x-1, y-7), IN(x-1, y-6), IN(x-1, y-5), IN(x-1, y-4), IN(x-1, y-3), IN(x-1, y-2), IN(x-1, y-1), IN(x-1, y), IN(x-1, y+1), IN(x-1, y+2), IN(x-1, y+3), IN(x-1, y+4), IN(x-1, y+5), IN(x-1, y+6), IN(x-1, y+7),
                    IN(x, y-7), IN(x, y-6), IN(x, y-5), IN(x, y-4), IN(x, y-3), IN(x, y-2), IN(x, y-1), IN(x, y), IN(x, y+1), IN(x, y+2), IN(x, y+3), IN(x, y+4), IN(x, y+5), IN(x, y+6), IN(x, y+7),
                    IN(x+1, y-7), IN(x+1, y-6), IN(x+1, y-5), IN(x+1, y-4), IN(x+1, y-3), IN(x+1, y-2), IN(x+1, y-1), IN(x+1, y), IN(x+1, y+1), IN(x+1, y+2), IN(x+1, y+3), IN(x+1, y+4), IN(x+1, y+5), IN(x+1, y+6), IN(x+1, y+7),
                    IN(x+2, y-7), IN(x+2, y-6), IN(x+2, y-5), IN(x+2, y-4), IN(x+2, y-3), IN(x+2, y-2), IN(x+2, y-1), IN(x+2, y), IN(x+2, y+1), IN(x+2, y+2), IN(x+2, y+3), IN(x+2, y+4), IN(x+2, y+5), IN(x+2, y+6), IN(x+2, y+7),
                    IN(x+3, y-7), IN(x+3, y-6), IN(x+3, y-5), IN(x+3, y-4), IN(x+3, y-3), IN(x+3, y-2), IN(x+3, y-1), IN(x+3, y), IN(x+3, y+1), IN(x+3, y+2), IN(x+3, y+3), IN(x+3, y+4), IN(x+3, y+5), IN(x+3, y+6), IN(x+3, y+7),
                    IN(x+4, y-7), IN(x+4, y-6), IN(x+4, y-5), IN(x+4, y-4), IN(x+4, y-3), IN(x+4, y-2), IN(x+4, y-1), IN(x+4, y), IN(x+4, y+1), IN(x+4, y+2), IN(x+4, y+3), IN(x+4, y+4), IN(x+4, y+5), IN(x+4, y+6), IN(x+4, y+7),
                    IN(x+5, y-7), IN(x+5, y-6), IN(x+5, y-5), IN(x+5, y-4), IN(x+5, y-3), IN(x+5, y-2), IN(x+5, y-1), IN(x+5, y), IN(x+5, y+1), IN(x+5, y+2), IN(x+5, y+3), IN(x+5, y+4), IN(x+5, y+5), IN(x+5, y+6), IN(x+5, y+7),
                    IN(x+6, y-7), IN(x+6, y-6), IN(x+6, y-5), IN(x+6, y-4), IN(x+6, y-3), IN(x+6, y-2), IN(x+6, y-1), IN(x+6, y), IN(x+6, y+1), IN(x+6, y+2), IN(x+6, y+3), IN(x+6, y+4), IN(x+6, y+5), IN(x+6, y+6), IN(x+6, y+7),
                    IN(x+7, y-7), IN(x+7, y-6), IN(x+7, y-5), IN(x+7, y-4), IN(x+7, y-3), IN(x+7, y-2), IN(x+7, y-1), IN(x+7, y), IN(x+7, y+1), IN(x+7, y+2), IN(x+7, y+3), IN(x+7, y+4), IN(x+7, y+5), IN(x+7, y+6), IN(x+7, y+7)
        };

        float currentPixel = IN(x, y);

        // bubble-sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                {
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        int mask = 0;
        if((currentPixel-v[vecSize/2]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2]*mask+currentPixel*(1-mask);
    }

}


__global__ void reomveOutliner3D15(int nx, int ny, int nz, int diff, float *d_out, float *d_in)
{
   // nx ny nz map to offset in the 1d array
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;

//    int offset = x+y* nx + ny * nx * z;
    if ((x < nx) && (y < ny) && (z < nz))
    {
        int winSize = 15;
        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;
        int newnx=toffset+nx;
        int newny=toffset+ny;

        x = x + loffset;
        y = y + loffset;

        int i = 0;

        for (int xx = x - loffset; xx <= x + roffset; xx++)
        {
            for (int yy = y - loffset; yy <= y + roffset; yy++)
            {
//                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries

                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
            }
        }

        // get the current pixel value
        // TODO get from local buffer instead of global memory

        float currentPixel = d_in[x+y*newnx+z*newnx*newny];

//        printf("the x is %d, y is %d, z is %d, current is %f, result is %f \n", x, y, z, currentPixel, v[vecSize/2] );

        // More optimize for the bubble sort
        for (int i = 0; i < vecSize; i++)
        {
            for (int j = i + 1; j < vecSize; j++)
            {
                if (v[i] > v[j])
                { /* swap? */
                    float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }

        // TODO more optimize for this part
        int mask = 0;
        if((currentPixel-v[vecSize/2]) >= diff)
            mask = 1;
        else
            mask = 0;



        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2]*mask+currentPixel*(1-mask);

    }

}


//#define s2(a,b)            { float tmp = a; a = min(a,b); b = max(tmp,b); }
//#define mn3(a,b,c)         s2(a,b); s2(a,c);
//#define mx3(a,b,c)         s2(b,c); s2(a,c);
//
//#define mnmx3(a,b,c)       mx3(a,b,c); s2(a,b);                               // 3 exchanges
//#define mnmx4(a,b,c,d)     s2(a,b); s2(c,d); s2(a,c); s2(b,d);                // 4 exchanges
//#define mnmx5(a,b,c,d,e)   s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e);          // 6 exchanges
//#define mnmx6(a,b,c,d,e,f) s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); // 7 exchanges
//
//#define SMEM(x,y)  smem[(x)+1][(y)+1]
//#define IN(x,y)    d_in[(y)*nx + (x)]
//
// __global__ void kernel(int nx, int ny, float *d_out, float *d_in, int size)
//{
//
//    int tx = threadIdx.x, ty = threadIdx.y;
//
//    // guards: is at boundary?
//    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X-1);
//    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y-1);
//
//    __shared__ float smem[BLOCK_X+2][BLOCK_Y+2];
//    // clear out shared memory (zero padding)
//    if (is_x_top)           SMEM(tx-1, ty  ) = 0;
//    else if (is_x_bot)      SMEM(tx+1, ty  ) = 0;
//    if (is_y_top) {         SMEM(tx  , ty-1) = 0;
//        if (is_x_top)       SMEM(tx-1, ty-1) = 0;
//        else if (is_x_bot)  SMEM(tx+1, ty-1) = 0;
//    } else if (is_y_bot) {  SMEM(tx  , ty+1) = 0;
//        if (is_x_top)       SMEM(tx-1, ty+1) = 0;
//        else if (is_x_bot)  SMEM(tx+1, ty+1) = 0;
//    }
//
//    // guards: is at boundary and still more image?
//    int x = blockIdx.x * blockDim.x + tx;
//    int y = blockIdx.y * blockDim.y + ty;
//    is_x_top &= (x > 0); is_x_bot &= (x < nx - 1);
//    is_y_top &= (y > 0); is_y_bot &= (y < ny - 1);
//
//    // each thread pulls from image
//                            SMEM(tx  , ty  ) = IN(x  , y  ); // self
//    if (is_x_top)           SMEM(tx-1, ty  ) = IN(x-1, y  );
//    else if (is_x_bot)      SMEM(tx+1, ty  ) = IN(x+1, y  );
//    if (is_y_top) {         SMEM(tx  , ty-1) = IN(x  , y-1);
//        if (is_x_top)       SMEM(tx-1, ty-1) = IN(x-1, y-1);
//        else if (is_x_bot)  SMEM(tx+1, ty-1) = IN(x+1, y-1);
//    } else if (is_y_bot) {  SMEM(tx  , ty+1) = IN(x  , y+1);
//        if (is_x_top)       SMEM(tx-1, ty+1) = IN(x-1, y+1);
//        else if (is_x_bot)  SMEM(tx+1, ty+1) = IN(x+1, y+1);
//    }
//    __syncthreads();
//
//    // pull top six from shared memory
//    float v[6] = { SMEM(tx-1, ty-1), SMEM(tx  , ty-1), SMEM(tx+1, ty-1),
//                   SMEM(tx-1, ty  ), SMEM(tx  , ty  ), SMEM(tx+1, ty  ) };
//
//    // with each pass, remove min and max values and add new value
//    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
//    v[5] = SMEM(tx-1, ty+1); // add new contestant
//    mnmx5(v[1], v[2], v[3], v[4], v[5]);
//    v[5] = SMEM(tx  , ty+1);
//    mnmx4(v[2], v[3], v[4], v[5]);
//    v[5] = SMEM(tx+1, ty+1);
//    mnmx3(v[3], v[4], v[5]);
//
////    printf("the x is %d, y is %d, result is %f \n", x, y, v[4] );
//
//    // pick the middle one
//    d_out[y*nx + x] = v[4];
//}
