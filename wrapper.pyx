import numpy as np
cimport numpy as np
import tomopy

#assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/medianFilter.hh":
    cdef cppclass C_mFilter "medianFilter":
        C_mFilter(int, int, int, int)
        void run2DFilter(int)
        void run2DRemoveOutliner(int, int)
        void retreive()
        void retreive_to(np.float32_t*)
        void setImage(np.float32_t*)

cdef class mFilter:
    cdef C_mFilter* g
    cdef int nx
    cdef int ny
    cdef int nz
    cdef int filterSize

    def __cinit__(self, int nx, int ny, int nz,  int filterSize):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.filterSize = filterSize
        self.g = new C_mFilter( self.nx, self.ny, self.nz, self.filterSize)
        #print self.nx, self.ny

    def run2DFilter(self, int size):
        self.g.run2DFilter(size)

    def run2DRemoveOutliner(self, int size, int diff):
        self.g.run2DRemoveOutliner(size, diff)

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] a = np.zeros(self.nx*self.ny, dtype=np.float32)
        # cdef np.ndarray[ndim=1, dtype=np.float32_t] a = np.zeros(self.nx*self.ny*self.nz, dtype=np.float32)
        self.g.retreive_to(&a[0])
        return a

    def setCuImage(self, np.ndarray[ndim=1, dtype=np.float32_t] arr):
        self.g.setImage(&arr[0])

def median_filter_cuda(arr, size=3):
    """
    Apply median filter to 3D array along 0 axis with GPU support.
    The winAllow 4 to 7 is for A6000, for Tian X support 3 to 8
    Parameters
    ----------
    arr : ndarray
        Input array.
    size : int, optional
        The size of the filter.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """

    winAllow = [4,5,6,7,15]

    if size in winAllow:
        prjsize = arr.shape[0]
        loffset = int(size/2)
        roffset = int((size-1)/2)
        imsizex =arr.shape[2] # image size for the input
        imsizey = arr.shape[1]

        filter = mFilter(imsizex, imsizey, prjsize, size)
        out = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

        for step in range (prjsize):
            im_noisecu=arr[step].astype(np.float32)
            im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
            im_noisecu = im_noisecu.flatten()

            filter.setCuImage(im_noisecu)
            filter.run2DFilter(size)
            results = filter.retreive()
            results=results.reshape(imsizey,imsizex)
            out[step]=results
    else:
        out = tomopy.median_filter(arr, size)

    return out

def remove_outlier_cuda(arr, dif, size=3):
    """
    Remove high intensity bright spots from a 3D array along specified
    dimension.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result.  If same as arr, process will be done in-place.


    Returns
    -------
    ndarray
       Corrected array.
    """
    # arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    winAllow = [4,5,6,7,15]

    if size in winAllow:
        prjsize = arr.shape[0]
        loffset = int(size/2)
        roffset = int((size-1)/2)
        imsizex =arr.shape[2] # image size for the input
        imsizey = arr.shape[1]

        filter = mFilter(imsizex, imsizey, prjsize, size)
        out = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

        for step in range (prjsize):
            im_noisecu=arr[step].astype(np.float32)
            im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
            im_noisecu = im_noisecu.flatten()

            filter.setCuImage(im_noisecu)
            filter.run2DRemoveOutliner(size, dif)
            results = filter.retreive()
            results=results.reshape(imsizey,imsizex)
            out[step]=results
    else:
        print("using cpu remove outlier")
        out = tomopy.remove_outlier(arr, dif, size)

    return out