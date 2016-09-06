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
#define INSE(X,Y) d_in[(X)+(Y)*(7+nx)] // 8
#define INEI(X,Y) d_in[(X)+(Y)*(8+nx)] // 9

#define INEL(X,Y) d_in[(X)+(Y)*(10+nx)] // 11

#define INTHI(X,Y) d_in[(X)+(Y)*(12+nx)] // 13


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

__global__ void kernel9ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 9;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[45]={ INEI(x-4, y-4),INEI(x-4, y-3), INEI(x-4, y-2), INEI(x-4, y-1), INEI(x-4, y), INEI(x-4, y+1), INEI(x-4, y+2), INEI(x-4, y+3),INEI(x-4, y+4),
                      INEI(x-3, y-4),INEI(x-3, y-3), INEI(x-3, y-2), INEI(x-3, y-1), INEI(x-3, y), INEI(x-3, y+1), INEI(x-3, y+2), INEI(x-3, y+3),INEI(x-3, y+4),
                      INEI(x-2, y-4),INEI(x-2, y-3), INEI(x-2, y-2), INEI(x-2, y-1), INEI(x-2, y), INEI(x-2, y+1), INEI(x-2, y+2), INEI(x-2, y+3),INEI(x-2, y+4),
                      INEI(x-1, y-4),INEI(x-1, y-3), INEI(x-1, y-2), INEI(x-1, y-1), INEI(x-1, y), INEI(x-1, y+1), INEI(x-1, y+2), INEI(x-1, y+3),INEI(x-1, y+4),
                      INEI(x, y-4),INEI(x, y-3), INEI(x, y-2), INEI(x, y-1), INEI(x, y), INEI(x, y+1), INEI(x, y+2), INEI(x, y+3),INEI(x, y+4)};



        const int ARR_SIZE = winSize*(winSize-winSize/2); // float array size

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
                v[0] = INEI(x+k, y+j); // change window size change here

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

__global__ void kernel11ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 11;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[66]={ INEL(x-5, y-5),INEL(x-5, y-4),INEL(x-5, y-3), INEL(x-5, y-2), INEL(x-5, y-1), INEL(x-5, y), INEL(x-5, y+1), INEL(x-5, y+2), INEL(x-5, y+3),INEL(x-5, y+4),INEL(x-5, y+5),
                      INEL(x-4, y-5),INEL(x-4, y-4),INEL(x-4, y-3), INEL(x-4, y-2), INEL(x-4, y-1), INEL(x-4, y), INEL(x-4, y+1), INEL(x-4, y+2), INEL(x-4, y+3),INEL(x-4, y+4),INEL(x-4, y+5),
                      INEL(x-3, y-5),INEL(x-3, y-4),INEL(x-3, y-3), INEL(x-3, y-2), INEL(x-3, y-1), INEL(x-3, y), INEL(x-3, y+1), INEL(x-3, y+2), INEL(x-3, y+3),INEL(x-3, y+4),INEL(x-3, y+5),
                      INEL(x-2, y-5),INEL(x-2, y-4),INEL(x-2, y-3), INEL(x-2, y-2), INEL(x-2, y-1), INEL(x-2, y), INEL(x-2, y+1), INEL(x-2, y+2), INEL(x-2, y+3),INEL(x-2, y+4),INEL(x-2, y+5),
                      INEL(x-1, y-5),INEL(x-1, y-4),INEL(x-1, y-3), INEL(x-1, y-2), INEL(x-1, y-1), INEL(x-1, y), INEL(x-1, y+1), INEL(x-1, y+2), INEL(x-1, y+3),INEL(x-1, y+4),INEL(x-1, y+5),
                      INEL(x, y-5),INEL(x, y-4),INEL(x, y-3), INEL(x, y-2), INEL(x, y-1), INEL(x, y), INEL(x, y+1), INEL(x, y+2), INEL(x, y+3),INEL(x, y+4),INEL(x, y+5)};



        const int ARR_SIZE = winSize*(winSize-winSize/2); // float array size

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
                v[0] = INEL(x+k, y+j); // change window size change here

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


__global__ void kernel13ME(int nx, int ny, float *d_out, float *d_in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < nx) && (y < ny))
    {
        int winSize = 13;

//        int vecSize = winSize*winSize;
        int loffset = winSize/2;
        int roffset = (winSize-1)/2;
//        int toffset = loffset+roffset;

        x = x + loffset;
        y = y + loffset;

// use macro to fetch the value, like loop unrolling
        float v[91]={ INTHI(x-6, y-6),INTHI(x-6, y-5),INTHI(x-6, y-4),INTHI(x-6, y-3), INTHI(x-6, y-2), INTHI(x-6, y-1), INTHI(x-6, y), INTHI(x-6, y+1), INTHI(x-6, y+2), INTHI(x-6, y+3),INTHI(x-6, y+4),INTHI(x-6, y+5),INTHI(x-6, y+6),
                      INTHI(x-5, y-6),INTHI(x-5, y-5),INTHI(x-5, y-4),INTHI(x-5, y-3), INTHI(x-5, y-2), INTHI(x-5, y-1), INTHI(x-5, y), INTHI(x-5, y+1), INTHI(x-5, y+2), INTHI(x-5, y+3),INTHI(x-5, y+4),INTHI(x-5, y+5),INTHI(x-5, y+6),
                      INTHI(x-4, y-6),INTHI(x-4, y-5),INTHI(x-4, y-4),INTHI(x-4, y-3), INTHI(x-4, y-2), INTHI(x-4, y-1), INTHI(x-4, y), INTHI(x-4, y+1), INTHI(x-4, y+2), INTHI(x-4, y+3),INTHI(x-4, y+4),INTHI(x-4, y+5),INTHI(x-4, y+6),
                      INTHI(x-3, y-6),INTHI(x-3, y-5),INTHI(x-3, y-4),INTHI(x-3, y-3), INTHI(x-3, y-2), INTHI(x-3, y-1), INTHI(x-3, y), INTHI(x-3, y+1), INTHI(x-3, y+2), INTHI(x-3, y+3),INTHI(x-3, y+4),INTHI(x-3, y+5),INTHI(x-3, y+6),
                      INTHI(x-2, y-6),INTHI(x-2, y-5),INTHI(x-2, y-4),INTHI(x-2, y-3), INTHI(x-2, y-2), INTHI(x-2, y-1), INTHI(x-2, y), INTHI(x-2, y+1), INTHI(x-2, y+2), INTHI(x-2, y+3),INTHI(x-2, y+4),INTHI(x-2, y+5),INTHI(x-2, y+6),
                      INTHI(x-1, y-6),INTHI(x-1, y-5),INTHI(x-1, y-4),INTHI(x-1, y-3), INTHI(x-1, y-2), INTHI(x-1, y-1), INTHI(x-1, y), INTHI(x-1, y+1), INTHI(x-1, y+2), INTHI(x-1, y+3),INTHI(x-1, y+4),INTHI(x-1, y+5),INTHI(x-1, y+6),
                      INTHI(x, y-6),INTHI(x, y-5),INTHI(x, y-4),INTHI(x, y-3), INTHI(x, y-2), INTHI(x, y-1), INTHI(x, y), INTHI(x, y+1), INTHI(x, y+2), INTHI(x, y+3),INTHI(x, y+4),INTHI(x, y+5),INTHI(x, y+6)};



        const int ARR_SIZE = winSize*(winSize-winSize/2); // float array size

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
                v[0] = INTHI(x+k, y+j); // change window size change here

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





