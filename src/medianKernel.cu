// Only support reflect mode right now


#include <medianFilter.hh>

//#define SMEM(X,Y)  smem[(X)+7][(Y)+7]
#define IN(X,Y)  d_in[(X)+(Y)*(14+nx)]

#define INO(X,Y) d_in[(X)+(Y)*(1+nx)]
#define INT(X,Y) d_in[(X)+(Y)*(2+nx)]
#define INTH(X,Y) d_in[(X)+(Y)*(3+nx)]
#define INF(X,Y) d_in[(X)+(Y)*(4+nx)]
#define INFI(X,Y) d_in[(X)+(Y)*(5+nx)]
#define INS(X,Y) d_in[(X)+(Y)*(6+nx)]
#define INSE(X,Y) d_in[(X)+(Y)*(7+nx)]

#define INZ(X,Y,Z)  d_in[(X)+(Y)*(14+nx)+(Z)*(14+nx)*ny]

#define swapd(a,b)    { float tmp = a; a = min(a,b); b = max(tmp,b); }

#define SMEM(x,y)  smem[(x)+1][(y)+1]

__global__ void kernel2ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 2;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[4]={INO(x-1, y-1), INO(x-1, y),
                    INO(x, y-1), INO(x,y)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INO(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

//        for(int k = 1; k <winSize/2; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < winSize/2; i++) {
//                swapd(v[i], v[winSize-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= winSize/2; i++) {
//                swapd(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = winSize-k-2; i >= winSize/2; i--) {
//                swapd(v[i], v[winSize-1-k]);
//            }
//        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[2];
    }

}

__global__ void kernel3ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 3;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[6]={INT(x-1, y-1), INT(x-1, y), INT(x-1, y+1),
                      INT(x, y-1), INT(x, y), INT(x, y+1)};


        const int ARR_SIZE = 6;

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= 1; k++) {

            for(int j = -1; j <= 1; j++) {

                // add new contestant to first position in array
                v[0] = INT(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k < 1; k++) {
            // move max/min into respective halves
            for(int i = k; i < 1; i++) {
                swapd(v[i], v[3-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= 1; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = 3-k-2; i >= 1; i--) {
                swapd(v[i], v[3-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[1];
    }

}

__global__ void kernel4ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 4;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[12]={INTH(x-2,y-2), INTH(x-2, y-1), INTH(x-2, y), INTH(x-2,y+1),
                    INTH(x-1,y-2), INTH(x-1, y-1), INTH(x-1, y), INTH(x-1,y+1),
                    INTH(x, y-2), INTH(x, y-1), INTH(x,y), INTH(x,y+1)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INTH(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize];
    }

}

__global__ void kernel5ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 5;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[15]={  INF(x-2, y-2), INF(x-2, y-1), INF(x-2, y), INF(x-2, y+1), INF(x-2, y+2),
                      INF(x-1, y-2), INF(x-1, y-1), INF(x-1, y), INF(x-1, y+1), INF(x-1, y+2),
                      INF(x, y-2), INF(x, y-1), INF(x, y), INF(x, y+1), INF(x, y+2)};


        const int ARR_SIZE = winSize*(winSize-winSize/2);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2; k++) {

            for(int j = -winSize/2; j <= winSize/2; j++) {

                // add new contestant to first position in array
                v[0] = INF(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k <winSize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < winSize/2; i++) {
                swapd(v[i], v[winSize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= winSize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = winSize-k-2; i >= winSize/2; i--) {
                swapd(v[i], v[winSize-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize/2];
    }

}

__global__ void kernel6ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 6;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[24]={INFI(x-3,y-3),INFI(x-3,y-2), INFI(x-3, y-1), INFI(x-3, y), INFI(x-3,y+1), INFI(x-3,y+2),
                    INFI(x-2,y-3),INFI(x-2,y-2), INFI(x-2, y-1), INFI(x-2, y), INFI(x-2,y+1), INFI(x-2,y+2),
                    INFI(x-1,y-3),INFI(x-1,y-2), INFI(x-1, y-1), INFI(x-1, y), INFI(x-1,y+1),INFI(x-1,y+2),
                    INFI(x, y-3), INFI(x, y-2), INFI(x, y-1), INFI(x,y), INFI(x,y+1), INFI(x,y+2)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INFI(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize];
    }

}


__global__ void kernel7ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 7;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[28]={ INS(x-3, y-3), INS(x-3, y-2), INS(x-3, y-1), INS(x-3, y), INS(x-3, y+1), INS(x-3, y+2), INS(x-3, y+3),
                      INS(x-2, y-3), INS(x-2, y-2), INS(x-2, y-1), INS(x-2, y), INS(x-2, y+1), INS(x-2, y+2), INS(x-2, y+3),
                      INS(x-1, y-3), INS(x-1, y-2), INS(x-1, y-1), INS(x-1, y), INS(x-1, y+1), INS(x-1, y+2), INS(x-1, y+3),
                      INS(x, y-3), INS(x, y-2), INS(x, y-1), INS(x, y), INS(x, y+1), INS(x, y+2), INS(x, y+3)};


        const int ARR_SIZE = winSize*(winSize-winSize/2);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2; k++) {

            for(int j = -winSize/2; j <= winSize/2; j++) {

                // add new contestant to first position in array
                v[0] = INS(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k <winSize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < winSize/2; i++) {
                swapd(v[i], v[winSize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= winSize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = winSize-k-2; i >= winSize/2; i--) {
                swapd(v[i], v[winSize-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize/2];
    }

}

__global__ void kernel8ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 8;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[40]={INSE(x-4,y-4),INSE(x-4,y-3),INSE(x-4,y-2), INSE(x-4, y-1), INSE(x-4, y), INSE(x-4,y+1), INSE(x-4,y+2),INSE(x-4,y+3),
                    INSE(x-3,y-4),INSE(x-3,y-3),INSE(x-3,y-2), INSE(x-3, y-1), INSE(x-3, y), INSE(x-3,y+1), INSE(x-3,y+2),INSE(x-3,y+3),
                    INSE(x-2,y-4),INSE(x-2,y-3),INSE(x-2,y-2), INSE(x-2, y-1), INSE(x-2, y), INSE(x-2,y+1), INSE(x-2,y+2),INSE(x-2,y+3),
                    INSE(x-1,y-4),INSE(x-1,y-3),INSE(x-1,y-2), INSE(x-1, y-1), INSE(x-1, y), INSE(x-1,y+1),INSE(x-1,y+2),INSE(x-1,y+3),
                    INSE(x, y-4),INSE(x, y-3), INSE(x, y-2), INSE(x, y-1), INSE(x,y), INSE(x,y+1), INSE(x,y+2), INSE(x,y+3)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INSE(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize];
    }

}


// Use the new exchange way

__global__ void kernel15ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 15;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[120]={IN(x-7, y-7), IN(x-7, y-6), IN(x-7, y-5), IN(x-7, y-4), IN(x-7, y-3), IN(x-7, y-2), IN(x-7, y-1), IN(x-7, y), IN(x-7, y+1), IN(x-7, y+2), IN(x-7, y+3), IN(x-7, y+4), IN(x-7, y+5), IN(x-7, y+6), IN(x-7, y+7),
                      IN(x-6, y-7), IN(x-6, y-6), IN(x-6, y-5), IN(x-6, y-4), IN(x-6, y-3), IN(x-6, y-2), IN(x-6, y-1), IN(x-6, y), IN(x-6, y+1), IN(x-6, y+2), IN(x-6, y+3), IN(x-6, y+4), IN(x-6, y+5), IN(x-6, y+6), IN(x-6, y+7),
                      IN(x-5, y-7), IN(x-5, y-6), IN(x-5, y-5), IN(x-5, y-4), IN(x-5, y-3), IN(x-5, y-2), IN(x-5, y-1), IN(x-5, y), IN(x-5, y+1), IN(x-5, y+2), IN(x-5, y+3), IN(x-5, y+4), IN(x-5, y+5), IN(x-5, y+6), IN(x-5, y+7),
                      IN(x-4, y-7), IN(x-4, y-6), IN(x-4, y-5), IN(x-4, y-4), IN(x-4, y-3), IN(x-4, y-2), IN(x-4, y-1), IN(x-4, y), IN(x-4, y+1), IN(x-4, y+2), IN(x-4, y+3), IN(x-4, y+4), IN(x-4, y+5), IN(x-4, y+6), IN(x-4, y+7),
                    IN(x-3, y-7), IN(x-3, y-6), IN(x-3, y-5), IN(x-3, y-4), IN(x-3, y-3), IN(x-3, y-2), IN(x-3, y-1), IN(x-3, y), IN(x-3, y+1), IN(x-3, y+2), IN(x-3, y+3), IN(x-3, y+4), IN(x-3, y+5), IN(x-3, y+6), IN(x-3, y+7),
                    IN(x-2, y-7), IN(x-2, y-6), IN(x-2, y-5), IN(x-2, y-4), IN(x-2, y-3), IN(x-2, y-2), IN(x-2, y-1), IN(x-2, y), IN(x-2, y+1), IN(x-2, y+2), IN(x-2, y+3), IN(x-2, y+4), IN(x-2, y+5), IN(x-2, y+6), IN(x-2, y+7),
                    IN(x-1, y-7), IN(x-1, y-6), IN(x-1, y-5), IN(x-1, y-4), IN(x-1, y-3), IN(x-1, y-2), IN(x-1, y-1), IN(x-1, y), IN(x-1, y+1), IN(x-1, y+2), IN(x-1, y+3), IN(x-1, y+4), IN(x-1, y+5), IN(x-1, y+6), IN(x-1, y+7),
                    IN(x, y-7), IN(x, y-6), IN(x, y-5), IN(x, y-4), IN(x, y-3), IN(x, y-2), IN(x, y-1), IN(x, y), IN(x, y+1), IN(x, y+2), IN(x, y+3), IN(x, y+4), IN(x, y+5), IN(x, y+6), IN(x, y+7)
                    };


        const int ARR_SIZE = 120;

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= 7; k++) {

            for(int j = -7; j <= 7; j++) {

                // add new contestant to first position in array
                v[0] = IN(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k < 7; k++) {
            // move max/min into respective halves
            for(int i = k; i < 7; i++) {
                swapd(v[i], v[15-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= 7; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = 15-k-2; i >= 7; i--) {
                swapd(v[i], v[15-1-k]);
            }
        }

        for(int k = 1; k < 7; k++) {
            // move max/min into respective halves
            for(int i = k; i < 7; i++) {
                swapd(v[i], v[15-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= 7; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = 15-k-2; i >= 7; i--) {
                swapd(v[i], v[15-1-k]);
            }
        }

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[7];
    }

}

__global__ void reomveOutliner2D2ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 2;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[4]={INO(x-1, y-1), INO(x-1, y),
                    INO(x, y-1), INO(x,y)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INO(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

//        for(int k = 1; k <winSize/2; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < winSize/2; i++) {
//                swapd(v[i], v[winSize-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= winSize/2; i++) {
//                swapd(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = winSize-k-2; i >= winSize/2; i--) {
//                swapd(v[i], v[winSize-1-k]);
//            }
//        }

        float currentPixel = INO(x, y);

        int mask = 0;
        if((currentPixel-v[winSize]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize]*mask+currentPixel*(1-mask);

    }

}


__global__ void reomveOutliner2D3ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 3;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[6]={INT(x-1, y-1), INT(x-1, y), INT(x-1, y+1),
                      INT(x, y-1), INT(x, y), INT(x, y+1)};


        const int ARR_SIZE = 6;

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= 1; k++) {

            for(int j = -1; j <= 1; j++) {

                // add new contestant to first position in array
                v[0] = INT(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        float currentPixel = INT(x, y);

        int mask = 0;
        if((currentPixel-v[1]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[1]*mask+currentPixel*(1-mask);
    }

}

__global__ void reomveOutliner2D4ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 4;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[12]={INTH(x-2,y-2), INTH(x-2, y-1), INTH(x-2, y), INTH(x-2,y+1),
                    INTH(x-1,y-2), INTH(x-1, y-1), INTH(x-1, y), INTH(x-1,y+1),
                    INTH(x, y-2), INTH(x, y-1), INTH(x,y), INTH(x,y+1)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INTH(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }

        float currentPixel = INTH(x, y);

        int mask = 0;
        if((currentPixel-v[winSize]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize]*mask+currentPixel*(1-mask);

    }

}


__global__ void reomveOutliner2D5ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 5;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[15]={  INF(x-2, y-2), INF(x-2, y-1), INF(x-2, y), INF(x-2, y+1), INF(x-2, y+2),
                      INF(x-1, y-2), INF(x-1, y-1), INF(x-1, y), INF(x-1, y+1), INF(x-1, y+2),
                      INF(x, y-2), INF(x, y-1), INF(x, y), INF(x, y+1), INF(x, y+2)};


        const int ARR_SIZE = winSize*(winSize-winSize/2);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2; k++) {

            for(int j = -winSize/2; j <= winSize/2; j++) {

                // add new contestant to first position in array
                v[0] = INF(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k <winSize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < winSize/2; i++) {
                swapd(v[i], v[winSize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= winSize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = winSize-k-2; i >= winSize/2; i--) {
                swapd(v[i], v[winSize-1-k]);
            }
        }

        float currentPixel = INF(x, y);

        int mask = 0;
        if((currentPixel-v[2]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[2]*mask+currentPixel*(1-mask);

    }

}


__global__ void reomveOutliner2D6ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 6;
//        float v[225] = {0};

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[24]={INFI(x-3,y-3),INFI(x-3,y-2), INFI(x-3, y-1), INFI(x-3, y), INFI(x-3,y+1), INFI(x-3,y+2),
                    INFI(x-2,y-3),INFI(x-2,y-2), INFI(x-2, y-1), INFI(x-2, y), INFI(x-2,y+1), INFI(x-2,y+2),
                    INFI(x-1,y-3),INFI(x-1,y-2), INFI(x-1, y-1), INFI(x-1, y), INFI(x-1,y+1),INFI(x-1,y+2),
                    INFI(x, y-3), INFI(x, y-2), INFI(x, y-1), INFI(x,y), INFI(x,y+1), INFI(x,y+2)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INFI(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }

        float currentPixel = INFI(x, y);

        int mask = 0;
        if((currentPixel-v[winSize]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize]*mask+currentPixel*(1-mask);
    }

}



__global__ void reomveOutliner2D7ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 7;

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[28]={ INS(x-3, y-3), INS(x-3, y-2), INS(x-3, y-1), INS(x-3, y), INS(x-3, y+1), INS(x-3, y+2), INS(x-3, y+3),
                      INS(x-2, y-3), INS(x-2, y-2), INS(x-2, y-1), INS(x-2, y), INS(x-2, y+1), INS(x-2, y+2), INS(x-2, y+3),
                      INS(x-1, y-3), INS(x-1, y-2), INS(x-1, y-1), INS(x-1, y), INS(x-1, y+1), INS(x-1, y+2), INS(x-1, y+3),
                      INS(x, y-3), INS(x, y-2), INS(x, y-1), INS(x, y), INS(x, y+1), INS(x, y+2), INS(x, y+3)};


        const int ARR_SIZE = winSize*(winSize-winSize/2);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2; k++) {

            for(int j = -winSize/2; j <= winSize/2; j++) {

                // add new contestant to first position in array
                v[0] = INS(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k <winSize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < winSize/2; i++) {
                swapd(v[i], v[winSize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= winSize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = winSize-k-2; i >= winSize/2; i--) {
                swapd(v[i], v[winSize-1-k]);
            }
        }

        float currentPixel = INS(x, y);

        int mask = 0;
        if((currentPixel-v[3]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[3]*mask+currentPixel*(1-mask);

    }

}


__global__ void reomveOutliner2D8ME(int nx, int ny, int diff, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 8;
//        float v[225] = {0};

        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[40]={INSE(x-4,y-4),INSE(x-4,y-3),INSE(x-4,y-2), INSE(x-4, y-1), INSE(x-4, y), INSE(x-4,y+1), INSE(x-4,y+2),INSE(x-4,y+3),
                    INSE(x-3,y-4),INSE(x-3,y-3),INSE(x-3,y-2), INSE(x-3, y-1), INSE(x-3, y), INSE(x-3,y+1), INSE(x-3,y+2),INSE(x-3,y+3),
                    INSE(x-2,y-4),INSE(x-2,y-3),INSE(x-2,y-2), INSE(x-2, y-1), INSE(x-2, y), INSE(x-2,y+1), INSE(x-2,y+2),INSE(x-2,y+3),
                    INSE(x-1,y-4),INSE(x-1,y-3),INSE(x-1,y-2), INSE(x-1, y-1), INSE(x-1, y), INSE(x-1,y+1),INSE(x-1,y+2),INSE(x-1,y+3),
                    INSE(x, y-4),INSE(x, y-3), INSE(x, y-2), INSE(x, y-1), INSE(x,y), INSE(x,y+1), INSE(x,y+2), INSE(x,y+3)};


        const int ARR_SIZE = winSize*(winSize/2+1);

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= winSize/2-1; k++) {

            for(int j = -winSize/2; j <= winSize/2-1; j++) {

                // add new contestant to first position in array
                v[0] = INSE(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        int fvecsize = 2*winSize;

        for(int k = 1; k <fvecsize/2; k++) {
            // move max/min into respective halves
            for(int i = k; i < fvecsize/2; i++) {
                swapd(v[i], v[fvecsize-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= fvecsize/2; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = fvecsize-k-2; i >= fvecsize/2; i--) {
                swapd(v[i], v[fvecsize-1-k]);
            }
        }


        float currentPixel = INSE(x, y);

        int mask = 0;
        if((currentPixel-v[winSize]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[winSize]*mask+currentPixel*(1-mask);

    }

}

__global__ void reomveOutliner2D15ME(int nx, int ny, int diff, float *d_out, float *d_in)
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
        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[120]={IN(x-7, y-7), IN(x-7, y-6), IN(x-7, y-5), IN(x-7, y-4), IN(x-7, y-3), IN(x-7, y-2), IN(x-7, y-1), IN(x-7, y), IN(x-7, y+1), IN(x-7, y+2), IN(x-7, y+3), IN(x-7, y+4), IN(x-7, y+5), IN(x-7, y+6), IN(x-7, y+7),
                      IN(x-6, y-7), IN(x-6, y-6), IN(x-6, y-5), IN(x-6, y-4), IN(x-6, y-3), IN(x-6, y-2), IN(x-6, y-1), IN(x-6, y), IN(x-6, y+1), IN(x-6, y+2), IN(x-6, y+3), IN(x-6, y+4), IN(x-6, y+5), IN(x-6, y+6), IN(x-6, y+7),
                      IN(x-5, y-7), IN(x-5, y-6), IN(x-5, y-5), IN(x-5, y-4), IN(x-5, y-3), IN(x-5, y-2), IN(x-5, y-1), IN(x-5, y), IN(x-5, y+1), IN(x-5, y+2), IN(x-5, y+3), IN(x-5, y+4), IN(x-5, y+5), IN(x-5, y+6), IN(x-5, y+7),
                      IN(x-4, y-7), IN(x-4, y-6), IN(x-4, y-5), IN(x-4, y-4), IN(x-4, y-3), IN(x-4, y-2), IN(x-4, y-1), IN(x-4, y), IN(x-4, y+1), IN(x-4, y+2), IN(x-4, y+3), IN(x-4, y+4), IN(x-4, y+5), IN(x-4, y+6), IN(x-4, y+7),
                    IN(x-3, y-7), IN(x-3, y-6), IN(x-3, y-5), IN(x-3, y-4), IN(x-3, y-3), IN(x-3, y-2), IN(x-3, y-1), IN(x-3, y), IN(x-3, y+1), IN(x-3, y+2), IN(x-3, y+3), IN(x-3, y+4), IN(x-3, y+5), IN(x-3, y+6), IN(x-3, y+7),
                    IN(x-2, y-7), IN(x-2, y-6), IN(x-2, y-5), IN(x-2, y-4), IN(x-2, y-3), IN(x-2, y-2), IN(x-2, y-1), IN(x-2, y), IN(x-2, y+1), IN(x-2, y+2), IN(x-2, y+3), IN(x-2, y+4), IN(x-2, y+5), IN(x-2, y+6), IN(x-2, y+7),
                    IN(x-1, y-7), IN(x-1, y-6), IN(x-1, y-5), IN(x-1, y-4), IN(x-1, y-3), IN(x-1, y-2), IN(x-1, y-1), IN(x-1, y), IN(x-1, y+1), IN(x-1, y+2), IN(x-1, y+3), IN(x-1, y+4), IN(x-1, y+5), IN(x-1, y+6), IN(x-1, y+7),
                    IN(x, y-7), IN(x, y-6), IN(x, y-5), IN(x, y-4), IN(x, y-3), IN(x, y-2), IN(x, y-1), IN(x, y), IN(x, y+1), IN(x, y+2), IN(x, y+3), IN(x, y+4), IN(x, y+5), IN(x, y+6), IN(x, y+7)
                    };


        const int ARR_SIZE = 120;

#pragma unroll
        for(int i = 0; i < ARR_SIZE/2; i++) {
            swapd(v[i], v[ARR_SIZE-1-i]);
        }

#pragma unroll
        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
            swapd(v[0], v[i]);
        }

#pragma unroll
        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swapd(v[i], v[ARR_SIZE-1]);
        }

        int last = ARR_SIZE-1;

        for(int k = 1; k <= 7; k++) {

            for(int j = -7; j <= 7; j++) {

                // add new contestant to first position in array
                v[0] = IN(x+k, y+j);

                last--;

                // place max in last half, min in first half
                for(int i = 0; i < (last+1)/2; i++) {
                    swapd(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(int i = 1; i <= last/2; i++) {
                    swapd(v[0], v[i]);
                }
                for(int i = last-1; i >= (last+1)/2; i--) {
                    swapd(v[i], v[last]);
                }
            }
        }

        for(int k = 1; k < 7; k++) {
            // move max/min into respective halves
            for(int i = k; i < 7; i++) {
                swapd(v[i], v[15-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= 7; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = 15-k-2; i >= 7; i--) {
                swapd(v[i], v[15-1-k]);
            }
        }

        for(int k = 1; k < 7; k++) {
            // move max/min into respective halves
            for(int i = k; i < 7; i++) {
                swapd(v[i], v[15-1-i]);
            }
            // move min into first pos
            for(int i = k+1; i <= 7; i++) {
                swapd(v[k], v[i]);
            }
            // move max into last pos
            for(int i = 15-k-2; i >= 7; i--) {
                swapd(v[i], v[15-1-k]);
            }
        }

        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[7];

        float currentPixel = IN(x, y);

        int mask = 0;
        if((currentPixel-v[7]) >= diff)
            mask = 1;
        else
            mask = 0;

        // pick the middle one
        d_out[(y-loffset)*nx + x-loffset] = v[7]*mask+currentPixel*(1-mask);

    }

}

//__global__ void kernel3MES(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int tx = threadIdx.x, ty = threadIdx.y;
//
//    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X-1);
//    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y-1);
//
//
//    __shared__ float smem[BLOCK_X+2][BLOCK_Y+2];
//
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
//
//    is_x_top &= (x > 0); is_x_bot &= (x < nx - 1);
//    is_y_top &= (y > 0); is_y_bot &= (y < ny - 1);
//
//    int winSize = 3;
//
//    int vecSize = winSize*winSize;
//    int loffset = winSize/2;
//    int roffset = (winSize-1)/2;
//    int toffset = loffset+roffset;
//
//    x = x + loffset;
//    y = y + loffset;
//
//
//                            SMEM(tx  , ty  ) = INT(x  , y  ); // self
//    if (is_x_top)           SMEM(tx-1, ty  ) = INT(x-1, y  );
//    else if (is_x_bot)      SMEM(tx+1, ty  ) = INT(x+1, y  );
//    if (is_y_top) {         SMEM(tx  , ty-1) = INT(x  , y-1);
//        if (is_x_top)       SMEM(tx-1, ty-1) = INT(x-1, y-1);
//        else if (is_x_bot)  SMEM(tx+1, ty-1) = INT(x+1, y-1);
//    } else if (is_y_bot) {  SMEM(tx  , ty+1) = INT(x  , y+1);
//        if (is_x_top)       SMEM(tx-1, ty+1) = INT(x-1, y+1);
//        else if (is_x_bot)  SMEM(tx+1, ty+1) = INT(x+1, y+1);
//    }
//    __syncthreads();
//
//    if ((x < nx) && (y < ny))
//    {
//
//// use macro to fetch the value, like loop unrolling
//        float v[6] = { SMEM(tx-1, ty-1), SMEM(tx  , ty-1), SMEM(tx+1, ty-1),
//                   SMEM(tx-1, ty  ), SMEM(tx  , ty  ), SMEM(tx+1, ty  ) };
//
//
//        const int ARR_SIZE = 6;
//
//#pragma unroll
//        for(int i = 0; i < ARR_SIZE/2; i++) {
//            swapd(v[i], v[ARR_SIZE-1-i]);
//        }
//
//#pragma unroll
//        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
//            swapd(v[0], v[i]);
//        }
//
//#pragma unroll
//        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
//            swapd(v[i], v[ARR_SIZE-1]);
//        }
//
//        int last = ARR_SIZE-1;
//
//        for(int k = 1; k <= 1; k++) {
//
//            for(int j = -1; j <= 1; j++) {
//
//                // add new contestant to first position in array
//                v[0] = SMEM(tx+j, ty+k);
//
//                last--;
//
//                // place max in last half, min in first half
//                for(int i = 0; i < (last+1)/2; i++) {
//                    swapd(v[i], v[last-i]);
//                }
//                // now perform swaps on each half such that
//                // max is in last pos, min is in first pos
//                for(int i = 1; i <= last/2; i++) {
//                    swapd(v[0], v[i]);
//                }
//                for(int i = last-1; i >= (last+1)/2; i--) {
//                    swapd(v[i], v[last]);
//                }
//            }
//        }
//
//        for(int k = 1; k < 1; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < 1; i++) {
//                swapd(v[i], v[3-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= 1; i++) {
//                swapd(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = 3-k-2; i >= 1; i--) {
//                swapd(v[i], v[3-1-k]);
//            }
//        }
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[1];
//    }
//
//}

// windows size 4 byb 4
//__global__ void kernel4(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int winSize = 4;
//    int loffset = winSize/2;
//    int roffset = winSize/2 - 1;
//    int toffset = loffset+roffset;
//
//    x = x + loffset;
//    y = y + loffset;
//
//    int i = 0;
//    float v[16] = {0};
//
//    for (int xx = x - loffset; xx <= x + roffset; xx++)
//    {
//        for (int yy = y - loffset; yy <= y + roffset; yy++)
//        {
//            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                v[i++] = d_in[yy*(nx+toffset) + xx];
//        }
//    }
//
//    // bubble-sort
//    for (int i = 0; i < 16; i++)
//    {
//        for (int j = i + 1; j < 16; j++)
//        {
//            if (v[i] > v[j])
//            { /* swap? */
//                float tmp = v[i];
//                v[i] = v[j];
//                v[j] = tmp;
//            }
//        }
//    }
//
//    // pick the middle one
//    d_out[(y-loffset)*nx + x-loffset] = v[8];
//}
//
//// Windows size 5 by b5
//__global__ void kernel5(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int winSize = 5;
//    float v[25] = {0};
//
//    int vecSize = winSize*winSize;
//    int loffset = winSize/2;
//    int roffset = (winSize-1)/2;
//    int toffset = loffset+roffset;
//
//    x = x + loffset;
//    y = y + loffset;
//
//    int i = 0;
//
//    for (int xx = x - loffset; xx <= x + roffset; xx++)
//    {
//        for (int yy = y - loffset; yy <= y + roffset; yy++)
//        {
//            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                v[i++] = d_in[yy*(nx+toffset) + xx];
//        }
//    }
//
//    // bubble-sort
//    for (int i = 0; i < vecSize; i++)
//    {
//        for (int j = i + 1; j < vecSize; j++)
//        {
//            if (v[i] > v[j])
//            { /* swap? */
//                float tmp = v[i];
//                v[i] = v[j];
//                v[j] = tmp;
//            }
//        }
//    }
//
//    // pick the middle one
//    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
//
//}
//
//// windows size 6 byb 6
//__global__ void kernel6(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int winSize = 6;
//    float v[36] = {0};
//
//    int vecSize = winSize*winSize;
//    int loffset = winSize/2;
//    int roffset = winSize/2 - 1;
//    int toffset = loffset+roffset;
//
//    x = x + loffset;
//    y = y + loffset;
//
//    int i = 0;
//
//
//    for (int xx = x - loffset; xx <= x + roffset; xx++)
//    {
//        for (int yy = y - loffset; yy <= y + roffset; yy++)
//        {
//            if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                v[i++] = d_in[yy*(nx+toffset) + xx];
//        }
//    }
//
//    // bubble-sort
//    for (int i = 0; i < vecSize; i++)
//    {
//        for (int j = i + 1; j < vecSize; j++)
//        {
//            if (v[i] > v[j])
//            { /* swap? */
//                float tmp = v[i];
//                v[i] = v[j];
//                v[j] = tmp;
//            }
//        }
//    }
//
//    // pick the middle one
//    d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
//}
//
//
//// window size 15 by b15
//__global__ void kernel15(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
////                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                    v[i++] = d_in[yy*(nx+toffset) + xx];
//            }
//        }
//
//        // bubble-sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
//    }
//
//}

//__global__ void kernel15M(int nx, int ny, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
////        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
//// use macro to fetch the value, like loop unrolling
//        float v[225]={IN(x-7, y-7), IN(x-7, y-6), IN(x-7, y-5), IN(x-7, y-4), IN(x-7, y-3), IN(x-7, y-2), IN(x-7, y-1), IN(x-7, y), IN(x-7, y+1), IN(x-7, y+2), IN(x-7, y+3), IN(x-7, y+4), IN(x-7, y+5), IN(x-7, y+6), IN(x-7, y+7),
//                      IN(x-6, y-7), IN(x-6, y-6), IN(x-6, y-5), IN(x-6, y-4), IN(x-6, y-3), IN(x-6, y-2), IN(x-6, y-1), IN(x-6, y), IN(x-6, y+1), IN(x-6, y+2), IN(x-6, y+3), IN(x-6, y+4), IN(x-6, y+5), IN(x-6, y+6), IN(x-6, y+7),
//                      IN(x-5, y-7), IN(x-5, y-6), IN(x-5, y-5), IN(x-5, y-4), IN(x-5, y-3), IN(x-5, y-2), IN(x-5, y-1), IN(x-5, y), IN(x-5, y+1), IN(x-5, y+2), IN(x-5, y+3), IN(x-5, y+4), IN(x-5, y+5), IN(x-5, y+6), IN(x-5, y+7),
//                      IN(x-4, y-7), IN(x-4, y-6), IN(x-4, y-5), IN(x-4, y-4), IN(x-4, y-3), IN(x-4, y-2), IN(x-4, y-1), IN(x-4, y), IN(x-4, y+1), IN(x-4, y+2), IN(x-4, y+3), IN(x-4, y+4), IN(x-4, y+5), IN(x-4, y+6), IN(x-4, y+7),
//                    IN(x-3, y-7), IN(x-3, y-6), IN(x-3, y-5), IN(x-3, y-4), IN(x-3, y-3), IN(x-3, y-2), IN(x-3, y-1), IN(x-3, y), IN(x-3, y+1), IN(x-3, y+2), IN(x-3, y+3), IN(x-3, y+4), IN(x-3, y+5), IN(x-3, y+6), IN(x-3, y+7),
//                    IN(x-2, y-7), IN(x-2, y-6), IN(x-2, y-5), IN(x-2, y-4), IN(x-2, y-3), IN(x-2, y-2), IN(x-2, y-1), IN(x-2, y), IN(x-2, y+1), IN(x-2, y+2), IN(x-2, y+3), IN(x-2, y+4), IN(x-2, y+5), IN(x-2, y+6), IN(x-2, y+7),
//                    IN(x-1, y-7), IN(x-1, y-6), IN(x-1, y-5), IN(x-1, y-4), IN(x-1, y-3), IN(x-1, y-2), IN(x-1, y-1), IN(x-1, y), IN(x-1, y+1), IN(x-1, y+2), IN(x-1, y+3), IN(x-1, y+4), IN(x-1, y+5), IN(x-1, y+6), IN(x-1, y+7),
//                    IN(x, y-7), IN(x, y-6), IN(x, y-5), IN(x, y-4), IN(x, y-3), IN(x, y-2), IN(x, y-1), IN(x, y), IN(x, y+1), IN(x, y+2), IN(x, y+3), IN(x, y+4), IN(x, y+5), IN(x, y+6), IN(x, y+7),
//                    IN(x+1, y-7), IN(x+1, y-6), IN(x+1, y-5), IN(x+1, y-4), IN(x+1, y-3), IN(x+1, y-2), IN(x+1, y-1), IN(x+1, y), IN(x+1, y+1), IN(x+1, y+2), IN(x+1, y+3), IN(x+1, y+4), IN(x+1, y+5), IN(x+1, y+6), IN(x+1, y+7),
//                    IN(x+2, y-7), IN(x+2, y-6), IN(x+2, y-5), IN(x+2, y-4), IN(x+2, y-3), IN(x+2, y-2), IN(x+2, y-1), IN(x+2, y), IN(x+2, y+1), IN(x+2, y+2), IN(x+2, y+3), IN(x+2, y+4), IN(x+2, y+5), IN(x+2, y+6), IN(x+2, y+7),
//                    IN(x+3, y-7), IN(x+3, y-6), IN(x+3, y-5), IN(x+3, y-4), IN(x+3, y-3), IN(x+3, y-2), IN(x+3, y-1), IN(x+3, y), IN(x+3, y+1), IN(x+3, y+2), IN(x+3, y+3), IN(x+3, y+4), IN(x+3, y+5), IN(x+3, y+6), IN(x+3, y+7),
//                    IN(x+4, y-7), IN(x+4, y-6), IN(x+4, y-5), IN(x+4, y-4), IN(x+4, y-3), IN(x+4, y-2), IN(x+4, y-1), IN(x+4, y), IN(x+4, y+1), IN(x+4, y+2), IN(x+4, y+3), IN(x+4, y+4), IN(x+4, y+5), IN(x+4, y+6), IN(x+4, y+7),
//                    IN(x+5, y-7), IN(x+5, y-6), IN(x+5, y-5), IN(x+5, y-4), IN(x+5, y-3), IN(x+5, y-2), IN(x+5, y-1), IN(x+5, y), IN(x+5, y+1), IN(x+5, y+2), IN(x+5, y+3), IN(x+5, y+4), IN(x+5, y+5), IN(x+5, y+6), IN(x+5, y+7),
//                    IN(x+6, y-7), IN(x+6, y-6), IN(x+6, y-5), IN(x+6, y-4), IN(x+6, y-3), IN(x+6, y-2), IN(x+6, y-1), IN(x+6, y), IN(x+6, y+1), IN(x+6, y+2), IN(x+6, y+3), IN(x+6, y+4), IN(x+6, y+5), IN(x+6, y+6), IN(x+6, y+7),
//                    IN(x+7, y-7), IN(x+7, y-6), IN(x+7, y-5), IN(x+7, y-4), IN(x+7, y-3), IN(x+7, y-2), IN(x+7, y-1), IN(x+7, y), IN(x+7, y+1), IN(x+7, y+2), IN(x+7, y+3), IN(x+7, y+4), IN(x+7, y+5), IN(x+7, y+6), IN(x+7, y+7)
//        };
//
//        // bubble-sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
//    }
//
//}


//__global__ void kernel15MS(int nx, int ny, float *d_out, float *d_in)
//{
//    __shared__ float smem[BLOCK_X+14][BLOCK_Y+14];
//
//    int tx = threadIdx.x, ty = threadIdx.y;
//
//    int x = blockIdx.x * blockDim.x + tx;
//    int y = blockIdx.y * blockDim.y + ty;
//
//    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X-1);
//    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y-1);
//
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
////        int roffset = (winSize-1)/2;
////        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        SMEM(tx , ty) = IN(x, y); // self pixel value
//
//
////        if (is_x_top)
////        {
////            SMEM(tx-1, ty) = IN(x-1, y);
////            SMEM(tx-2, ty) = IN(x-2, y);
////            SMEM(tx-3, ty) = IN(x-3, y);
////            SMEM(tx-4, ty) = IN(x-4, y);
////            SMEM(tx-5, ty) = IN(x-5, y);
////            SMEM(tx-6, ty) = IN(x-6, y);
////            SMEM(tx-7, ty) = IN(x-7, y);
////        }
////        else if (is_x_bot)
////        {
////            SMEM(tx+1, ty) = IN(x+1, y);
////            SMEM(tx+2, ty) = IN(x+2, y);
////            SMEM(tx+3, ty) = IN(x+3, y);
////            SMEM(tx+4, ty) = IN(x+4, y);
////            SMEM(tx+5, ty) = IN(x+5, y);
////            SMEM(tx+6, ty) = IN(x+6, y);
////            SMEM(tx+7, ty) = IN(x+7, y);
////        }
////
////        if (is_y_top)
////        {
////            SMEM(tx, ty-1) = IN(x, y-1);
////            SMEM(tx, ty-2) = IN(x, y-2);
////            SMEM(tx, ty-3) = IN(x, y-3);
////            SMEM(tx, ty-4) = IN(x, y-4);
////            SMEM(tx, ty-5) = IN(x, y-5);
////            SMEM(tx, ty-6) = IN(x, y-6);
////            SMEM(tx, ty-7) = IN(x, y-7);
////            if (is_x_top)
////            {
////                SMEM(tx-1, ty-1) = IN(x-1, y-1);
////                SMEM(tx-2, ty-2) = IN(x-2, y-2);
////                SMEM(tx-3, ty-3) = IN(x-3, y-3);
////                SMEM(tx-4, ty-4) = IN(x-4, y-4);
////                SMEM(tx-5, ty-5) = IN(x-5, y-5);
////                SMEM(tx-6, ty-6) = IN(x-6, y-6);
////                SMEM(tx-7, ty-7) = IN(x-7, y-7);
////
//////                SMEM(tx-1, ty-1) = IN(x-1, y-1);
////
////            }
////            else if (is_x_bot)
////            {
////                SMEM(tx+1, ty-1) = IN(x+1, y-1);
////                SMEM(tx+2, ty-2) = IN(x+2, y-2);
////                SMEM(tx+3, ty-3) = IN(x+3, y-3);
////                SMEM(tx+4, ty-4) = IN(x+4, y-4);
////                SMEM(tx+5, ty-5) = IN(x+5, y-5);
////                SMEM(tx+6, ty-6) = IN(x+6, y-6);
////                SMEM(tx+7, ty-7) = IN(x+7, y-7);
////            }
////        }
////        else if (is_y_bot)
////        {
////            SMEM(tx, ty+1) = IN(x, y+1);
////            SMEM(tx, ty+2) = IN(x, y+2);
////            SMEM(tx, ty+3) = IN(x, y+3);
////            SMEM(tx, ty+4) = IN(x, y+4);
////            SMEM(tx, ty+5) = IN(x, y+5);
////            SMEM(tx, ty+6) = IN(x, y+6);
////            SMEM(tx, ty+7) = IN(x, y+7);
////            if (is_x_top)
////            {
////                SMEM(tx-1, ty+1) = IN(x-1, y+1);
////                SMEM(tx-2, ty+2) = IN(x-2, y+2);
////                SMEM(tx-3, ty+3) = IN(x-3, y+3);
////                SMEM(tx-4, ty+4) = IN(x-4, y+4);
////                SMEM(tx-5, ty+5) = IN(x-5, y+5);
////                SMEM(tx-6, ty+6) = IN(x-6, y+6);
////                SMEM(tx-7, ty+7) = IN(x-7, y+7);
////            }
////            else if (is_x_bot)
////            {
////                SMEM(tx+1, ty+1) = IN(x+1, y+1);
////                SMEM(tx+2, ty+2) = IN(x+2, y+2);
////                SMEM(tx+3, ty+3) = IN(x+3, y+3);
////                SMEM(tx+4, ty+4) = IN(x+4, y+4);
////                SMEM(tx+5, ty+5) = IN(x+5, y+5);
////                SMEM(tx+6, ty+6) = IN(x+6, y+6);
////                SMEM(tx+7, ty+7) = IN(x+7, y+7);
////            }
//
////        }
//    __syncthreads();
////    printf("the x is %d, y is %d, i is , result is %f\n", x, y, SMEM(tx-7, ty-7));
//    // use macro to fetch the value, like loop unrolling
//    float v[225]={SMEM(tx-7, ty-7), SMEM(tx-7, ty-6), SMEM(tx-7, ty-5), SMEM(tx-7, ty-4), SMEM(tx-7, ty-3), SMEM(tx-7, ty-2), SMEM(tx-7, ty-1), SMEM(tx-7, ty), SMEM(tx-7, ty+1), SMEM(tx-7, ty+2), SMEM(tx-7, ty+3), SMEM(tx-7, ty+4), SMEM(tx-7, ty+5), SMEM(tx-7, ty+6), SMEM(tx-7, ty+7),
//                  SMEM(tx-6, ty-7), SMEM(tx-6, ty-6), SMEM(tx-6, ty-5), SMEM(tx-6, ty-4), SMEM(tx-6, ty-3), SMEM(tx-6, ty-2), SMEM(tx-6, ty-1), SMEM(tx-6, ty), SMEM(tx-6, ty+1), SMEM(tx-6, ty+2), SMEM(tx-6, ty+3), SMEM(tx-6, ty+4), SMEM(tx-6, ty+5), SMEM(tx-6, ty+6), SMEM(tx-6, ty+7),
//                  SMEM(tx-5, ty-7), SMEM(tx-5, ty-6), SMEM(tx-5, ty-5), SMEM(tx-5, ty-4), SMEM(tx-5, ty-3), SMEM(tx-5, ty-2), SMEM(tx-5, ty-1), SMEM(tx-5, ty), SMEM(tx-5, ty+1), SMEM(tx-5, ty+2), SMEM(tx-5, ty+3), SMEM(tx-5, ty+4), SMEM(tx-5, ty+5), SMEM(tx-5, ty+6), SMEM(tx-5, ty+7),
//                  SMEM(tx-4, ty-7), SMEM(tx-4, ty-6), SMEM(tx-4, ty-5), SMEM(tx-4, ty-4), SMEM(tx-4, ty-3), SMEM(tx-4, ty-2), SMEM(tx-4, ty-1), SMEM(tx-4, ty), SMEM(tx-4, ty+1), SMEM(tx-4, ty+2), SMEM(tx-4, ty+3), SMEM(tx-4, ty+4), SMEM(tx-4, ty+5), SMEM(tx-4, ty+6), SMEM(tx-4, ty+7),
//                SMEM(tx-3, ty-7), SMEM(tx-3, ty-6), SMEM(tx-3, ty-5), SMEM(tx-3, ty-4), SMEM(tx-3, ty-3), SMEM(tx-3, ty-2), SMEM(tx-3, ty-1), SMEM(tx-3, ty), SMEM(tx-3, ty+1), SMEM(tx-3, ty+2), SMEM(tx-3, ty+3), SMEM(tx-3, ty+4), SMEM(tx-3, ty+5), SMEM(tx-3, ty+6), SMEM(tx-3, ty+7),
//                SMEM(tx-2, ty-7), SMEM(tx-2, ty-6), SMEM(tx-2, ty-5), SMEM(tx-2, ty-4), SMEM(tx-2, ty-3), SMEM(tx-2, ty-2), SMEM(tx-2, ty-1), SMEM(tx-2, ty), SMEM(tx-2, ty+1), SMEM(tx-2, ty+2), SMEM(tx-2, ty+3), SMEM(tx-2, ty+4), SMEM(tx-2, ty+5), SMEM(tx-2, ty+6), SMEM(tx-2, ty+7),
//                SMEM(tx-1, ty-7), SMEM(tx-1, ty-6), SMEM(tx-1, ty-5), SMEM(tx-1, ty-4), SMEM(tx-1, ty-3), SMEM(tx-1, ty-2), SMEM(tx-1, ty-1), SMEM(tx-1, ty), SMEM(tx-1, ty+1), SMEM(tx-1, ty+2), SMEM(tx-1, ty+3), SMEM(tx-1, ty+4), SMEM(tx-1, ty+5), SMEM(tx-1, ty+6), SMEM(tx-1, ty+7),
//                SMEM(tx, ty-7), SMEM(tx, ty-6), SMEM(tx, ty-5), SMEM(tx, ty-4), SMEM(tx, ty-3), SMEM(tx, ty-2), SMEM(tx, ty-1), SMEM(tx, ty), SMEM(tx, ty+1), SMEM(tx, ty+2), SMEM(tx, ty+3), SMEM(tx, ty+4), SMEM(tx, ty+5), SMEM(tx, ty+6), SMEM(tx, ty+7),
//                SMEM(tx+1, ty-7), SMEM(tx+1, ty-6), SMEM(tx+1, ty-5), SMEM(tx+1, ty-4), SMEM(tx+1, ty-3), SMEM(tx+1, ty-2), SMEM(tx+1, ty-1), SMEM(tx+1, ty), SMEM(tx+1, ty+1), SMEM(tx+1, ty+2), SMEM(tx+1, ty+3), SMEM(tx+1, ty+4), SMEM(tx+1, ty+5), SMEM(tx+1, ty+6), SMEM(tx+1, ty+7),
//                SMEM(tx+2, ty-7), SMEM(tx+2, ty-6), SMEM(tx+2, ty-5), SMEM(tx+2, ty-4), SMEM(tx+2, ty-3), SMEM(tx+2, ty-2), SMEM(tx+2, ty-1), SMEM(tx+2, ty), SMEM(tx+2, ty+1), SMEM(tx+2, ty+2), SMEM(tx+2, ty+3), SMEM(tx+2, ty+4), SMEM(tx+2, ty+5), SMEM(tx+2, ty+6), SMEM(tx+2, ty+7),
//                SMEM(tx+3, ty-7), SMEM(tx+3, ty-6), SMEM(tx+3, ty-5), SMEM(tx+3, ty-4), SMEM(tx+3, ty-3), SMEM(tx+3, ty-2), SMEM(tx+3, ty-1), SMEM(tx+3, ty), SMEM(tx+3, ty+1), SMEM(tx+3, ty+2), SMEM(tx+3, ty+3), SMEM(tx+3, ty+4), SMEM(tx+3, ty+5), SMEM(tx+3, ty+6), SMEM(tx+3, ty+7),
//                SMEM(tx+4, ty-7), SMEM(tx+4, ty-6), SMEM(tx+4, ty-5), SMEM(tx+4, ty-4), SMEM(tx+4, ty-3), SMEM(tx+4, ty-2), SMEM(tx+4, ty-1), SMEM(tx+4, ty), SMEM(tx+4, ty+1), SMEM(tx+4, ty+2), SMEM(tx+4, ty+3), SMEM(tx+4, ty+4), SMEM(tx+4, ty+5), SMEM(tx+4, ty+6), SMEM(tx+4, ty+7),
//                SMEM(tx+5, ty-7), SMEM(tx+5, ty-6), SMEM(tx+5, ty-5), SMEM(tx+5, ty-4), SMEM(tx+5, ty-3), SMEM(tx+5, ty-2), SMEM(tx+5, ty-1), SMEM(tx+5, ty), SMEM(tx+5, ty+1), SMEM(tx+5, ty+2), SMEM(tx+5, ty+3), SMEM(tx+5, ty+4), SMEM(tx+5, ty+5), SMEM(tx+5, ty+6), SMEM(tx+5, ty+7),
//                SMEM(tx+6, ty-7), SMEM(tx+6, ty-6), SMEM(tx+6, ty-5), SMEM(tx+6, ty-4), SMEM(tx+6, ty-3), SMEM(tx+6, ty-2), SMEM(tx+6, ty-1), SMEM(tx+6, ty), SMEM(tx+6, ty+1), SMEM(tx+6, ty+2), SMEM(tx+6, ty+3), SMEM(tx+6, ty+4), SMEM(tx+6, ty+5), SMEM(tx+6, ty+6), SMEM(tx+6, ty+7),
//                SMEM(tx+7, ty-7), SMEM(tx+7, ty-6), SMEM(tx+7, ty-5), SMEM(tx+7, ty-4), SMEM(tx+7, ty-3), SMEM(tx+7, ty-2), SMEM(tx+7, ty-1), SMEM(tx+7, ty), SMEM(tx+7, ty+1), SMEM(tx+7, ty+2), SMEM(tx+7, ty+3), SMEM(tx+7, ty+4), SMEM(tx+7, ty+5), SMEM(tx+7, ty+6), SMEM(tx+7, ty+7)
//    };
//
//
//        // bubble-sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2];
//    }
//
//}
//
//__global__ void kernel3D2(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        int winSize = 2;
//        float v[4] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//        int newnx=toffset+nx;
//        int newny=toffset+ny;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
////                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
//            }
//        }
//
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];
//
//    }
//
//}

//__global__ void kernel3D15(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        // initial the window size, the local vector size
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2; // the left and top offset
//        int roffset = (winSize-1)/2; // the right and bottom offset
//        int toffset = loffset+roffset; // the overall offset
//
//// The new x' y' is the plus offset
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//        // Put the neighbour pixel into the local memory for the later bubble sort
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
//                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//                    // find the read address of the x y z pixel
//                    v[i++] = d_in[xx+yy*(nx+toffset)+z*(nx+toffset)*(ny+toffset)];
//            }
//        }
//
//        // do the bubble sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                {   // bubble sort
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        //    printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );
//        // put the final result value to the output array
//        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];
//
//    }
//
//}
//
//__global__ void kernel3D15XZ(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        // initial the window size, the local vector size
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2; // the left and top offset
//        int roffset = (winSize-1)/2; // the right and bottom offset
//        int toffset = loffset+roffset; // the overall offset
//
//// The new x' y' is the plus offset
//        x = x + loffset;
////        y = y + loffset;
//        z = z + loffset;
//
//
//        int i = 0;
//        // Put the neighbour pixel into the local memory for the later bubble sort
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int zz = z - loffset; zz <= z + roffset; zz++)
//            {
////                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//                    // find the read address of the x y z pixel
//                    v[i++] = d_in[xx+y*(nx+toffset)+zz*(nx+toffset)*ny];
//            }
//        }
//
//        // do the bubble sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                {   // bubble sort
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
////          printf("the x is %d, y is %d, z is %d, nz is %d  result is %f \n", x, y, z, nz, v[vecSize/2] );
//        // put the final result value to the output array
//        d_out[x-loffset + (z-loffset)*nx + y*nx*nz ] = v[vecSize/2];
//
//    }
//
//}
//
//__global__ void kernel3D15XZME(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        // initial the window size, the local vector size
//        int winSize = 15;
//        float v1[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2; // the left and top offset
//        int roffset = (winSize-1)/2; // the right and bottom offset
//        int toffset = loffset+roffset; // the overall offset
//
//// The new x' y' is the plus offset
//        x = x + loffset;
////        y = y + loffset;
//        z = z + loffset;
//
//// use macro to fetch the value, like loop unrolling
//
//        float v[120]={INZ(x-7,y, z-7), INZ(x-7,y, z-6), INZ(x-7,y, z-5), INZ(x-7,y, z-4), INZ(x-7,y, z-3), INZ(x-7,y, z-2), INZ(x-7,y, z-1), INZ(x-7,y, z), INZ(x-7,y, z+1), INZ(x-7,y, z+2), INZ(x-7,y, z+3), INZ(x-7,y, z+4), INZ(x-7,y, z+5), INZ(x-7,y, z+6), INZ(x-7,y, z+7),
//                      INZ(x-6,y, z-7), INZ(x-6,y, z-6), INZ(x-6,y, z-5), INZ(x-6,y, z-4), INZ(x-6,y, z-3), INZ(x-6,y, z-2), INZ(x-6,y, z-1), INZ(x-6,y, z), INZ(x-6,y, z+1), INZ(x-6,y, z+2), INZ(x-6,y, z+3), INZ(x-6,y, z+4), INZ(x-6,y, z+5), INZ(x-6,y, z+6), INZ(x-6,y, z+7),
//                      INZ(x-5,y, z-7), INZ(x-5,y, z-6), INZ(x-5,y, z-5), INZ(x-5,y, z-4), INZ(x-5,y, z-3), INZ(x-5,y, z-2), INZ(x-5,y, z-1), INZ(x-5,y, z), INZ(x-5,y, z+1), INZ(x-5,y, z+2), INZ(x-5,y, z+3), INZ(x-5,y, z+4), INZ(x-5,y, z+5), INZ(x-5,y, z+6), INZ(x-5,y, z+7),
//                      INZ(x-4,y, z-7), INZ(x-4,y, z-6), INZ(x-4,y, z-5), INZ(x-4,y, z-4), INZ(x-4,y, z-3), INZ(x-4,y, z-2), INZ(x-4,y, z-1), INZ(x-4,y, z), INZ(x-4,y, z+1), INZ(x-4,y, z+2), INZ(x-4,y, z+3), INZ(x-4,y, z+4), INZ(x-4,y, z+5), INZ(x-4,y, z+6), INZ(x-4,y, z+7),
//                    INZ(x-3,y, z-7), INZ(x-3,y, z-6), INZ(x-3,y, z-5), INZ(x-3,y, z-4), INZ(x-3,y, z-3), INZ(x-3,y, z-2), INZ(x-3, y,z-1), INZ(x-3, y,z), INZ(x-3,y, z+1), INZ(x-3,y, z+2), INZ(x-3,y, z+3), INZ(x-3,y, z+4), INZ(x-3,y, z+5), INZ(x-3,y, z+6), INZ(x-3,y, z+7),
//                    INZ(x-2, y,z-7), INZ(x-2,y, z-6), INZ(x-2, y,z-5), INZ(x-2, y,z-4), INZ(x-2,y, z-3), INZ(x-2,y, z-2), INZ(x-2,y, z-1), INZ(x-2,y, z), INZ(x-2,y, z+1), INZ(x-2,y, z+2), INZ(x-2,y, z+3), INZ(x-2,y, z+4), INZ(x-2,y, z+5), INZ(x-2,y, z+6), INZ(x-2,y, z+7),
//                    INZ(x-1,y, z-7), INZ(x-1,y, z-6), INZ(x-1,y, z-5), INZ(x-1,y, z-4), INZ(x-1,y, z-3), INZ(x-1, y,z-2), INZ(x-1,y, z-1), INZ(x-1,y, z), INZ(x-1,y, z+1), INZ(x-1,y, z+2), INZ(x-1,y, z+3), INZ(x-1,y, z+4), INZ(x-1,y, z+5), INZ(x-1,y, z+6), INZ(x-1,y, z+7),
//                    INZ(x, y,z-7), INZ(x, y,z-6), INZ(x,y, z-5), INZ(x,y, z-4), INZ(x, y,z-3), INZ(x,y, z-2), INZ(x,y, z-1), INZ(x,y, z), INZ(x, y,z+1), INZ(x,y, z+2), INZ(x,y, z+3), INZ(x,y, z+4), INZ(x,y, z+5), INZ(x,y, z+6), INZ(x,y, z+7)
//                    };
//
////        int i = 0;
////        // Put the neighbour pixel into the local memory for the later bubble sort
////        for (int xx = x - loffset; xx <= x + roffset; xx++)
////        {
////            for (int zz = z - loffset; zz <= z + roffset; zz++)
////            {
////                    v1[i++] = d_in[xx+y*(nx+toffset)+zz*(nx+toffset)*ny];
////            }
////        }
//
//
////        for (int i = 0; i< 120; i++)
////        {
////
////                printf("the x is %d, y is %d, z is %d, v is %f, v[1] is %f \n", x, y, z, v[i], v1[i] );
////                if(v[i] != v1[i])
////                    printf("the false \n");
////
////        }
//        // do the bubble sort
////        for (int i = 0; i < vecSize; i++)
////        {
////            for (int j = i + 1; j < vecSize; j++)
////            {
////                if (v[i] > v[j])
////                {   // bubble sort
////                    float tmp = v[i];
////                    v[i] = v[j];
////                    v[j] = tmp;
////                }
////            }
////        }
//
//
//        const int ARR_SIZE = 120;
//
//#pragma unroll
//        for(int i = 0; i < ARR_SIZE/2; i++) {
//            swapd(v[i], v[ARR_SIZE-1-i]);
//        }
//
//#pragma unroll
//        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
//            swapd(v[0], v[i]);
//        }
//
//#pragma unroll
//        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
//            swapd(v[i], v[ARR_SIZE-1]);
//        }
//
//        int last = ARR_SIZE-1;
//
//        for(int k = 1; k <= 7; k++) {
//
//            for(int j = -7; j <= 7; j++) {
//
//                // add new contestant to first position in array
//                v[0] = INZ(x+k, y, z+j);
//
////                IN(x+k, y+j);
//
//                last--;
//
//                // place max in last half, min in first half
//                for(int i = 0; i < (last+1)/2; i++) {
//                    swapd(v[i], v[last-i]);
//                }
//                // now perform swaps on each half such that
//                // max is in last pos, min is in first pos
//                for(int i = 1; i <= last/2; i++) {
//                    swapd(v[0], v[i]);
//                }
//                for(int i = last-1; i >= (last+1)/2; i--) {
//                    swapd(v[i], v[last]);
//                }
//            }
//        }
//
//        for(int k = 1; k < 7; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < 7; i++) {
//                swapd(v[i], v[15-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= 7; i++) {
//                swapd(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = 15-k-2; i >= 7; i--) {
//                swapd(v[i], v[15-1-k]);
//            }
//        }
//
//        for(int k = 1; k < 7; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < 7; i++) {
//                swapd(v[i], v[15-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= 7; i++) {
//                swapd(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = 15-k-2; i >= 7; i--) {
//                swapd(v[i], v[15-1-k]);
//            }
//        }
//
//
////          printf("the x is %d, y is %d, z is %d, nz is %d  result is %f \n", x, y, z, nz, v[vecSize/2] );
//        // put the final result value to the output array
//        d_out[x-loffset + (z-loffset)*nx + y*nx*nz ] = v[7];
//
//    }
//
//}
//
//__global__ void kernelLool3D15(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
////    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//
//
//        for(int z = 0; z < nz; z++)
//        {
//            i = 0;
//
//            for (int xx = x - loffset; xx <= x + roffset; xx++)
//            {
//                for (int yy = y - loffset; yy <= y + roffset; yy++)
//                {
//
//                    v[i++] = d_in[xx+yy*(nx+toffset)+z*(nx+toffset)*(ny+toffset)];
//                }
//            }
//
//            for (int i = 0; i < vecSize; i++)
//            {
//                for (int j = i + 1; j < vecSize; j++)
//                {
//                    if (v[i] > v[j])
//                    { /* swap? */
//                        float tmp = v[i];
//                        v[i] = v[j];
//                        v[j] = tmp;
//                    }
//                }
//            }
//
////            printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );
//
//            d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];
//
//        }
//
//    }
//
//}
//
//__global__ void kernelLool3D15XZY(int nx, int ny, int nz,  float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int z = blockIdx.y*blockDim.y + threadIdx.y;
////    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (z < nz))
//    {
//        int winSize = 15;
//        float v[225]={0};
//
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//
//        int newnx = nx+toffset;
//        int newny = ny+toffset;
//        int zoffset = z*newnx*newny;
//
//        x = x + loffset;
////        y = y + loffset;
////        z = z + loffset;
//
//        int i = 0;
//
//        for(int y = loffset; y < ny+loffset; y++)
//        {
//            i = 0;
//
////            for (int xx = x - loffset; xx <= x + roffset; xx++)
////            {
////                for (int yy = y - loffset; yy <= y + roffset; yy++)
////                {
////
////                    v[i++] = d_in[xx+yy*newnx+zoffset];
////                }
////            }
//
//
//
//            for (int i = 0; i < vecSize; i++)
//            {
//                for (int j = i + 1; j < vecSize; j++)
//                {
//                    if (v[i] > v[j])
//                    {
//                        float tmp = v[i];
//                        v[i] = v[j];
//                        v[j] = tmp;
//                    }
//                }
//            }
//
////            printf("the x is %d, y is %d, z is %d, result is %f \n", x, y, z, v[vecSize/2] );
//
//            d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2];
//
//        }
//
//    }
//
//}
//
//__global__ void reomveOutliner3D2(int nx, int ny, int nz, int diff, float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        int winSize = 2;
//        float v[4] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//        int newnx=toffset+nx;
//        int newny=toffset+ny;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
////                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
//            }
//        }
//
//        // get the current pixel value
//        // TODO get from local buffer instead of global memory
//
//        float currentPixel = d_in[x+y*newnx+z*newnx*newny];
//
//
//
//        // More optimize for the bubble sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        // TODO more optimize for this part
//        int mask = 0;
//        float realdiff = currentPixel-v[vecSize/2];
//        printf("the x is %d, y is %d, z is %d, current is %f, result is %f \n", x, y, z, currentPixel, v[vecSize/2] );
//
//        if( realdiff >= diff)
//            mask = 1;
//        else
//            mask = 0;
//
//
//
//        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2]*mask+currentPixel*(1-mask);
//
//    }
//
//}

//__global__ void reomveOutliner2D15(int nx, int ny, int diff, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
//                v[i++] = d_in[yy*(nx+toffset) + xx];
//            }
//        }
//
//        float currentPixel = IN(x, y);
//
//        // bubble-sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                {
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        int mask = 0;
//        if((currentPixel-v[vecSize/2]) >= diff)
//            mask = 1;
//        else
//            mask = 0;
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2]*mask+currentPixel*(1-mask);
//    }
//
//}



//__global__ void reomveOutliner2D15M(int nx, int ny, int diff, float *d_out, float *d_in)
//{
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if ((x < nx) && (y < ny))
//    {
//        int winSize = 15;
////        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
////        int toffset = loffset+roffset;
//
//        x = x + loffset;
//        y = y + loffset;
//
////        int i = 0;
//
//        float v[225]={IN(x-7, y-7), IN(x-7, y-6), IN(x-7, y-5), IN(x-7, y-4), IN(x-7, y-3), IN(x-7, y-2), IN(x-7, y-1), IN(x-7, y), IN(x-7, y+1), IN(x-7, y+2), IN(x-7, y+3), IN(x-7, y+4), IN(x-7, y+5), IN(x-7, y+6), IN(x-7, y+7),
//                      IN(x-6, y-7), IN(x-6, y-6), IN(x-6, y-5), IN(x-6, y-4), IN(x-6, y-3), IN(x-6, y-2), IN(x-6, y-1), IN(x-6, y), IN(x-6, y+1), IN(x-6, y+2), IN(x-6, y+3), IN(x-6, y+4), IN(x-6, y+5), IN(x-6, y+6), IN(x-6, y+7),
//                      IN(x-5, y-7), IN(x-5, y-6), IN(x-5, y-5), IN(x-5, y-4), IN(x-5, y-3), IN(x-5, y-2), IN(x-5, y-1), IN(x-5, y), IN(x-5, y+1), IN(x-5, y+2), IN(x-5, y+3), IN(x-5, y+4), IN(x-5, y+5), IN(x-5, y+6), IN(x-5, y+7),
//                      IN(x-4, y-7), IN(x-4, y-6), IN(x-4, y-5), IN(x-4, y-4), IN(x-4, y-3), IN(x-4, y-2), IN(x-4, y-1), IN(x-4, y), IN(x-4, y+1), IN(x-4, y+2), IN(x-4, y+3), IN(x-4, y+4), IN(x-4, y+5), IN(x-4, y+6), IN(x-4, y+7),
//                    IN(x-3, y-7), IN(x-3, y-6), IN(x-3, y-5), IN(x-3, y-4), IN(x-3, y-3), IN(x-3, y-2), IN(x-3, y-1), IN(x-3, y), IN(x-3, y+1), IN(x-3, y+2), IN(x-3, y+3), IN(x-3, y+4), IN(x-3, y+5), IN(x-3, y+6), IN(x-3, y+7),
//                    IN(x-2, y-7), IN(x-2, y-6), IN(x-2, y-5), IN(x-2, y-4), IN(x-2, y-3), IN(x-2, y-2), IN(x-2, y-1), IN(x-2, y), IN(x-2, y+1), IN(x-2, y+2), IN(x-2, y+3), IN(x-2, y+4), IN(x-2, y+5), IN(x-2, y+6), IN(x-2, y+7),
//                    IN(x-1, y-7), IN(x-1, y-6), IN(x-1, y-5), IN(x-1, y-4), IN(x-1, y-3), IN(x-1, y-2), IN(x-1, y-1), IN(x-1, y), IN(x-1, y+1), IN(x-1, y+2), IN(x-1, y+3), IN(x-1, y+4), IN(x-1, y+5), IN(x-1, y+6), IN(x-1, y+7),
//                    IN(x, y-7), IN(x, y-6), IN(x, y-5), IN(x, y-4), IN(x, y-3), IN(x, y-2), IN(x, y-1), IN(x, y), IN(x, y+1), IN(x, y+2), IN(x, y+3), IN(x, y+4), IN(x, y+5), IN(x, y+6), IN(x, y+7),
//                    IN(x+1, y-7), IN(x+1, y-6), IN(x+1, y-5), IN(x+1, y-4), IN(x+1, y-3), IN(x+1, y-2), IN(x+1, y-1), IN(x+1, y), IN(x+1, y+1), IN(x+1, y+2), IN(x+1, y+3), IN(x+1, y+4), IN(x+1, y+5), IN(x+1, y+6), IN(x+1, y+7),
//                    IN(x+2, y-7), IN(x+2, y-6), IN(x+2, y-5), IN(x+2, y-4), IN(x+2, y-3), IN(x+2, y-2), IN(x+2, y-1), IN(x+2, y), IN(x+2, y+1), IN(x+2, y+2), IN(x+2, y+3), IN(x+2, y+4), IN(x+2, y+5), IN(x+2, y+6), IN(x+2, y+7),
//                    IN(x+3, y-7), IN(x+3, y-6), IN(x+3, y-5), IN(x+3, y-4), IN(x+3, y-3), IN(x+3, y-2), IN(x+3, y-1), IN(x+3, y), IN(x+3, y+1), IN(x+3, y+2), IN(x+3, y+3), IN(x+3, y+4), IN(x+3, y+5), IN(x+3, y+6), IN(x+3, y+7),
//                    IN(x+4, y-7), IN(x+4, y-6), IN(x+4, y-5), IN(x+4, y-4), IN(x+4, y-3), IN(x+4, y-2), IN(x+4, y-1), IN(x+4, y), IN(x+4, y+1), IN(x+4, y+2), IN(x+4, y+3), IN(x+4, y+4), IN(x+4, y+5), IN(x+4, y+6), IN(x+4, y+7),
//                    IN(x+5, y-7), IN(x+5, y-6), IN(x+5, y-5), IN(x+5, y-4), IN(x+5, y-3), IN(x+5, y-2), IN(x+5, y-1), IN(x+5, y), IN(x+5, y+1), IN(x+5, y+2), IN(x+5, y+3), IN(x+5, y+4), IN(x+5, y+5), IN(x+5, y+6), IN(x+5, y+7),
//                    IN(x+6, y-7), IN(x+6, y-6), IN(x+6, y-5), IN(x+6, y-4), IN(x+6, y-3), IN(x+6, y-2), IN(x+6, y-1), IN(x+6, y), IN(x+6, y+1), IN(x+6, y+2), IN(x+6, y+3), IN(x+6, y+4), IN(x+6, y+5), IN(x+6, y+6), IN(x+6, y+7),
//                    IN(x+7, y-7), IN(x+7, y-6), IN(x+7, y-5), IN(x+7, y-4), IN(x+7, y-3), IN(x+7, y-2), IN(x+7, y-1), IN(x+7, y), IN(x+7, y+1), IN(x+7, y+2), IN(x+7, y+3), IN(x+7, y+4), IN(x+7, y+5), IN(x+7, y+6), IN(x+7, y+7)
//        };
//
//        float currentPixel = IN(x, y);
//
//        // bubble-sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                {
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        int mask = 0;
//        if((currentPixel-v[vecSize/2]) >= diff)
//            mask = 1;
//        else
//            mask = 0;
//
//        // pick the middle one
//        d_out[(y-loffset)*nx + x-loffset] = v[vecSize/2]*mask+currentPixel*(1-mask);
//    }
//
//}

//
//
//
//__global__ void reomveOutliner3D15(int nx, int ny, int nz, int diff, float *d_out, float *d_in)
//{
//   // nx ny nz map to offset in the 1d array
//    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
//    unsigned z = blockIdx.z*blockDim.z + threadIdx.z;
//
////    int offset = x+y* nx + ny * nx * z;
//    if ((x < nx) && (y < ny) && (z < nz))
//    {
//        int winSize = 15;
//        float v[225] = {0};
//
//        int vecSize = winSize*winSize;
//        int loffset = winSize/2;
//        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;
//        int newnx=toffset+nx;
//        int newny=toffset+ny;
//
//        x = x + loffset;
//        y = y + loffset;
//
//        int i = 0;
//
//        for (int xx = x - loffset; xx <= x + roffset; xx++)
//        {
//            for (int yy = y - loffset; yy <= y + roffset; yy++)
//            {
////                if (0 <= xx && xx < nx+toffset && 0 <= yy && yy < ny+toffset) // boundaries
//
//                    v[i++] = d_in[xx+yy*newnx+z*newnx*newny];
//            }
//        }
//
//        // get the current pixel value
//        // TODO get from local buffer instead of global memory
//
//        float currentPixel = d_in[x+y*newnx+z*newnx*newny];
//
////        printf("the x is %d, y is %d, z is %d, current is %f, result is %f \n", x, y, z, currentPixel, v[vecSize/2] );
//
//        // More optimize for the bubble sort
//        for (int i = 0; i < vecSize; i++)
//        {
//            for (int j = i + 1; j < vecSize; j++)
//            {
//                if (v[i] > v[j])
//                { /* swap? */
//                    float tmp = v[i];
//                    v[i] = v[j];
//                    v[j] = tmp;
//                }
//            }
//        }
//
//        // TODO more optimize for this part
//        int mask = 0;
//        if((currentPixel-v[vecSize/2]) >= diff)
//            mask = 1;
//        else
//            mask = 0;
//
//
//
//        d_out[x-loffset + (y-loffset)*nx + z*nx*ny ] = v[vecSize/2]*mask+currentPixel*(1-mask);
//
//    }
//
//}



