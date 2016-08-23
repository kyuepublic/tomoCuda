import numpy as np
cimport numpy as np

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
