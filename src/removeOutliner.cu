//
//static const int THREADS_X = 16;
//static const int THREADS_Y = 16;
//
//#define __DH__ __device__ __host__
//
//typedef long long   dim_t;
//
//// Exchange trick: Morgan McGuire, ShaderX 2008
//#define swap(a,b)    { T tmp = a; a = min(a,b); b = max(tmp,b); }
//
//typedef enum {
//    ///
//    /// Out of bound values are 0
//    ///
//    AF_PAD_ZERO = 0,
//
//    ///
//    /// Out of bound values are symmetric over the edge
//    ///
//    AF_PAD_SYM
//} af_border_type;
//
//
//template<typename T>
//struct Param
//{
//    T *ptr;
//    dim_t dims[4];
//    dim_t strides[4];
//};
//
//template<typename T>
//class CParam
//{
//public:
//    const T *ptr;
//    dim_t dims[4];
//    dim_t strides[4];
//
//    __DH__ CParam(const T *iptr, const dim_t *idims, const dim_t *istrides) :
//        ptr(iptr)
//    {
//        for (int i = 0; i < 4; i++) {
//            dims[i] = idims[i];
//            strides[i] = istrides[i];
//        }
//    }
//
//    __DH__ CParam(Param<T> &in) : ptr(in.ptr)
//    {
//        for (int i = 0; i < 4; i++) {
//            dims[i] = in.dims[i];
//            strides[i] = in.strides[i];
//        }
//    }
//
//    __DH__ ~CParam() {}
//};
//
//__forceinline__ __device__
//int lIdx(int x, int y, int stride1, int stride0)
//{
//    return (y*stride1 + x*stride0);
//}
//
//template<typename T, af_border_type pad>
//__device__
//void load2ShrdMem(T * shrd, const T * in,
//                  int lx, int ly, int shrdStride,
//                  int dim0, int dim1,
//                  int gx, int gy,
//                  int inStride1, int inStride0)
//{
//    switch(pad) {
//        case AF_PAD_ZERO:
//            {
//                if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
//                    shrd[lIdx(lx, ly, shrdStride, 1)] = T(0);
//                else
//                    shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
//            }
//            break;
//        case AF_PAD_SYM:
//            {
//                if (gx<0) gx *= -1;
//                if (gy<0) gy *= -1;
//                if (gx>=dim0) gx = 2*(dim0-1) - gx;
//                if (gy>=dim1) gy = 2*(dim1-1) - gy;
//
//                shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
//            }
//            break;
//    }
//}
//
//template<typename T, af_border_type pad, unsigned w_len, unsigned w_wid>
//__global__
//void medfilt(Param<T> out, CParam<T> in, int nBBS0, int nBBS1)
//{
//    __shared__ T shrdMem[(THREADS_X+w_len-1)*(THREADS_Y+w_wid-1)];
//
//    // calculate necessary offset and window parameters
//    const int padding = w_len-1;
//    const int halo    = padding/2;
//    const int shrdLen = blockDim.x + padding;
//
//    // batch offsets
//    unsigned b2 = blockIdx.x / nBBS0;
//    unsigned b3 = blockIdx.y / nBBS1;
//    const T* iptr    = (const T *) in.ptr + (b2 *  in.strides[2] + b3 *  in.strides[3]);
//    T*       optr    = (T *      )out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);
//
//    // local neighborhood indices
//    int lx = threadIdx.x;
//    int ly = threadIdx.y;
//
//    // global indices
//    int gx = blockDim.x * (blockIdx.x-b2*nBBS0) + lx;
//    int gy = blockDim.y * (blockIdx.y-b3*nBBS1) + ly;
//
//    // pull image to local memory
//    for (int b=ly, gy2=gy; b<shrdLen; b+=blockDim.y, gy2+=blockDim.y) {
//        // move row_set get_local_size(1) along coloumns
//        for (int a=lx, gx2=gx; a<shrdLen; a+=blockDim.x, gx2+=blockDim.x) {
//            load2ShrdMem<T, pad>(shrdMem, iptr, a, b, shrdLen, in.dims[0], in.dims[1],
//                    gx2-halo, gy2-halo, in.strides[1], in.strides[0]);
//        }
//    }
//
//    __syncthreads();
//
//    // Only continue if we're at a valid location
//    if (gx < in.dims[0] && gy < in.dims[1]) {
//
//        const int ARR_SIZE = w_len * (w_wid-w_wid/2);
//        // pull top half from shared memory into local memory
//        T v[ARR_SIZE];
//#pragma unroll
//        for(int k = 0; k <= w_wid/2; k++) {
//#pragma unroll
//            for(int i = 0; i < w_len; i++) {
//                v[w_len*k + i] = shrdMem[lIdx(lx+i,ly+k,shrdLen,1)];
//            }
//        }
//
//        // with each pass, remove min and max values and add new value
//        // initial sort
//        // ensure min in first half, max in second half
//#pragma unroll
//        for(int i = 0; i < ARR_SIZE/2; i++) {
//            swap(v[i], v[ARR_SIZE-1-i]);
//        }
//        // move min in first half to first pos
//#pragma unroll
//        for(int i = 1; i < (ARR_SIZE+1)/2; i++) {
//            swap(v[0], v[i]);
//        }
//        // move max in second half to last pos
//#pragma unroll
//        for(int i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
//            swap(v[i], v[ARR_SIZE-1]);
//        }
//
//        int last = ARR_SIZE-1;
//
//        for(int k = 1+w_wid/2; k < w_wid; k++) {
//
//            for(int j = 0; j < w_len; j++) {
//
//                // add new contestant to first position in array
//                v[0] = shrdMem[lIdx(lx+j, ly+k, shrdLen, 1)];
//
//                last--;
//
//                // place max in last half, min in first half
//                for(int i = 0; i < (last+1)/2; i++) {
//                    swap(v[i], v[last-i]);
//                }
//                // now perform swaps on each half such that
//                // max is in last pos, min is in first pos
//                for(int i = 1; i <= last/2; i++) {
//                    swap(v[0], v[i]);
//                }
//                for(int i = last-1; i >= (last+1)/2; i--) {
//                    swap(v[i], v[last]);
//                }
//            }
//        }
//
//        // no more new contestants
//        // may still have to sort the last row
//        // each outer loop drops the min and max
//        for(int k = 1; k < w_len/2; k++) {
//            // move max/min into respective halves
//            for(int i = k; i < w_len/2; i++) {
//                swap(v[i], v[w_len-1-i]);
//            }
//            // move min into first pos
//            for(int i = k+1; i <= w_len/2; i++) {
//                swap(v[k], v[i]);
//            }
//            // move max into last pos
//            for(int i = w_len-k-2; i >= w_len/2; i--) {
//                swap(v[i], v[w_len-1-k]);
//            }
//        }
//
//        // pick the middle element of the first row
//        optr[gy*out.strides[1]+gx*out.strides[0]] = v[w_len/2];
//    }
//}