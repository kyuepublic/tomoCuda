import numpy as np
cimport numpy as np

#assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/medianFilter.hh":
    cdef cppclass C_mFilter "medianFilter":
        C_mFilter(int, int, int, int)
        void run3DFilter(int)
        void run2DFilter(int)
        void run2DLoopFilter(int)
        void run3DRemoveOutliner(int, int)
        void run2DLoopFilterXZY(int)
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

    def run3DFilter(self, int size):
        self.g.run3DFilter(size)

    def run2DFilter(self, int size):
        self.g.run2DFilter(size)

    def run2DLoopFilter(self, int size):
        self.g.run2DLoopFilter(size)

    def run3DRemoveOutliner(self, int size, int diff):
        self.g.run3DRemoveOutliner(size, diff)

    def run2DLoopFilterXZY(self, int size):
        self.g.run2DLoopFilterXZY(size)

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] a = np.zeros(self.nx*self.ny*self.nz, dtype=np.float32)

        self.g.retreive_to(&a[0])
        return a

    def setCuImage(self, np.ndarray[ndim=1, dtype=np.float32_t] arr):
        self.g.setImage(&arr[0])
