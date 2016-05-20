// Only support reflect mode right now

#include <stdio.h>
#include <medianFilter.hh>


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
