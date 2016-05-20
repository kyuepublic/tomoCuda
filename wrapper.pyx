import numpy as np
cimport numpy as np

#assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/medianFilter.hh":
    cdef cppclass C_mFilter "medianFilter":
        C_mFilter(np.float32_t*, int, int, int)
        void runFilter(int)
        void retreive()
        void retreive_to(np.float32_t*)

cdef class mFilter:
    cdef C_mFilter* g
    cdef int nx
    cdef int ny
    cdef int filterSize

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t] arr, int nx, int ny, int filterSize):
        self.nx = nx
        self.ny = ny
        self.filterSize = filterSize
        self.g = new C_mFilter(&arr[0], self.nx, self.ny, self.filterSize)
        #print self.nx, self.ny

    def runFilter(self, int size):
        self.g.runFilter(size)

    def retreive_inplace(self):
        self.g.retreive()
    # cython type float32_t
    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] a = np.zeros(self.nx*self.ny, dtype=np.float32)

        self.g.retreive_to(&a[0])

        return a